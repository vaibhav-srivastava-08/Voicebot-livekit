"""Loads and searches the MSIL car sales knowledge base ("MSIL KNOWLEDGE BASE/" directory).

The KB is a mix of hand-authored HTML (model pages, Q&A articles, variant/colour dumps),
Markdown (price list), and plain text (policy/escalation notes). Everything is parsed into
plain-text chunks once at import time (so the parse cost is paid once per worker process, not
per call - same prewarming principle as agent.main's VAD loading).

RETRIEVAL PIPELINE
    1. BM25 keyword search (unchanged from before) and a local ONNX multilingual sentence
       embedding model (fastembed - pure onnxruntime, no torch, no network calls once the
       model is baked into the image at build time; see Dockerfile.agent) each independently
       rank all chunks for the query.
    2. The two rankings are combined via Reciprocal Rank Fusion (RRF): simple, doesn't need
       the retrievers' raw scores to be on comparable scales (BM25 scores and cosine
       similarities aren't), and degrades gracefully to BM25-only if the embedding model
       failed to load for any reason.
    3. A lightweight exact-keyword-overlap re-rank pass runs over the fused top candidates -
       not a learned/cross-encoder re-rank, intentionally simple, no extra model or network.

HINDI/HINGLISH CODE-SWITCHING
    Empirically, the embedding model's cross-lingual similarity for pairs like "price" and
    its Hindi equivalent "keemat" is real but only moderate (~0.35-0.5 cosine similarity,
    measured directly) - not reliable enough to depend on alone, since our entire KB corpus is
    in English. So a small explicit Hindi/Hinglish -> English synonym map guarantees this
    behavior deterministically for the terms customers actually use, rather than hoping the
    model's multilingual training happens to bridge it.

STRUCTURED FACTS
    Price-list bullets and per-variant spec lines are additionally parsed into small
    {variant, price, specs, ...} records alongside their prose chunk. search() surfaces any
    such records among its results in a clearly separated, one-fact-per-line block, so the LLM
    can't blend two different variants' numbers together when composing its reply.

Chunk text (and structured field values) stay RAW (numbers as digits/symbols where the source
has them): converting that into spoken English words is SYSTEM_PROMPT's job (Hardcoded Rule
Two), not this module's.
"""

import hashlib
import re
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi

from shared.logger import get_logger

logger = get_logger(__name__)

KB_DIR = Path(__file__).parent.parent / "MSIL KNOWLEDGE BASE"

# Multilingual, ONNX-only (no torch), ~220MB - small/fast enough to bake into the agent image
# and run on CPU with no network calls. See Dockerfile.agent for the build-time bake step.
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_CACHE_ROOT = Path(__file__).parent.parent / ".fastembed_cache"
_MODEL_CACHE_DIR = _CACHE_ROOT / "models"
_EMBEDDINGS_CACHE_FILE = _CACHE_ROOT / "kb_chunk_embeddings.npz"

_HEADING_TAGS = {"h1", "h2", "h3", "h4"}
_MIN_CHUNK_LENGTH = 15
_RRF_K = 60  # standard damping constant from the Reciprocal Rank Fusion literature

Chunk = tuple[str, "dict[str, str] | None"]


# ---------------------------------------------------------------------------------------------
# Structured fact extraction
# ---------------------------------------------------------------------------------------------


def _split_on_last_colon(text: str) -> tuple[str, str] | None:
    """Split "{left}: {right}" on the LAST colon - handles both a single "Label: value" line
    and a "Variant, specs: price" bullet, where the value/price is unambiguously the tail."""
    if ":" not in text:
        return None
    left, _, right = text.rpartition(":")
    left, right = left.strip(), right.strip().rstrip(".")
    if not left or not right:
        return None
    return left, right


def _structured_from_attribute_line(text: str, model: str) -> dict[str, str] | None:
    """A model page's <p> attribute line ("Engine: 1.5 liter...") -> {model, field, value}."""
    parsed = _split_on_last_colon(text)
    if parsed is None:
        return None
    field, value = parsed
    return {"model": model, "field": field, "value": value}


def _structured_from_variant_price_line(text: str) -> dict[str, str] | None:
    """A price-list bullet ("Variant, specs: price") -> {variant, specs, price}. The variant
    name is self-identifying (prefixed with its model, e.g. "Brezza LXi MT"), so no separate
    model tag is needed here."""
    parsed = _split_on_last_colon(text)
    if parsed is None:
        return None
    prefix, price = parsed
    variant, _, specs = prefix.partition(",")
    variant = variant.strip()
    if not variant:
        return None
    result = {"variant": variant, "price": price}
    if specs.strip():
        result["specs"] = specs.strip()
    return result


def _structured_from_variant_feature_block(text: str) -> dict[str, str] | None:
    """A variants_features.html <p> ("Variant: X. Mileage: Y. Key features: Z.") -> whatever
    labeled fields are present, keyed by lowercased label."""
    fields: dict[str, str] = {}
    for piece in text.split(". "):
        parsed = _split_on_last_colon(piece)
        if parsed:
            label, value = parsed
            fields[label.lower()] = value
    return fields if "variant" in fields else None


# ---------------------------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------------------------


def _chunk_html(path: Path) -> list[Chunk]:
    """Split one KB HTML file into topic-sized (text, structured_fact_or_None) chunks,
    following whichever of the corpus's layouts the file uses: <article>-wrapped Q&A blocks,
    heading-delimited pages with one fact per <p>/<li> (model pages, variant/feature
    listings), or a <pre> plain-text dump split on "MODEL:" markers.

    Each <p>/<li> becomes its OWN chunk (tagged with the nearest heading for context) rather
    than merging everything under a heading into one blob: a model page's price, engine,
    mileage, features, and competitor comparisons are all siblings under a single heading, and
    merging them diluted term-density enough that BM25 ranked short, unrelated Q&A snippets
    above the actual price paragraph for a plain "<model> price" query.
    """
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    body = soup.body or soup
    title = soup.title.get_text(strip=True) if soup.title else path.stem
    is_variant_feature_file = "variants_features" in path.stem

    chunks: list[Chunk] = []
    current_heading = title

    def emit(text: str, structured: dict[str, str] | None = None) -> None:
        text = text.strip()
        if len(text) >= _MIN_CHUNK_LENGTH:
            prefix = title if current_heading == title else f"{title} - {current_heading}"
            chunks.append((f"{prefix}: {text}", structured))

    for child in body.find_all(recursive=False):
        name = getattr(child, "name", None)
        if name == "article":
            emit(child.get_text(separator=" ", strip=True))
        elif name in _HEADING_TAGS:
            current_heading = child.get_text(strip=True)
        elif name == "pre":
            for block in re.split(r"\n(?=MODEL:)", child.get_text().strip()):
                emit(block)
        elif name == "ul":
            for li in child.find_all("li", recursive=False):
                text = li.get_text(separator=" ", strip=True)
                emit(text, _structured_from_variant_price_line(text))
        elif name is not None:
            text = child.get_text(separator=" ", strip=True)
            if is_variant_feature_file:
                structured = _structured_from_variant_feature_block(text)
            elif current_heading != title:
                structured = _structured_from_attribute_line(text, model=current_heading)
            else:
                structured = None
            emit(text, structured)
    return chunks


def _chunk_price_markdown(path: Path) -> list[Chunk]:
    """Split the ex-showroom price list into one chunk per model summary sentence, plus one
    chunk per variant price bullet (each with its {variant, specs, price} extracted)."""
    text = path.read_text(encoding="utf-8").strip()
    chunks: list[Chunk] = []

    sections = re.split(r"\n(?=## )", text)[1:]  # [0] is the doc-level intro, not a model
    for section in sections:
        lines = section.strip().splitlines()
        model_heading = lines[0].removeprefix("## ").strip()
        for line in lines[1:]:
            line = line.strip()
            if not line or line == "---":
                continue
            if line.startswith("- "):
                bullet = line[2:].strip()
                chunks.append(
                    (f"{model_heading}: {bullet}", _structured_from_variant_price_line(bullet))
                )
            elif len(line) >= _MIN_CHUNK_LENGTH:
                chunks.append((f"{model_heading}: {line}", None))
    return chunks


def _chunk_delimited_text(path: Path, heading_prefix: str) -> list[Chunk]:
    """Split a plain-text KB file (policy/escalation notes) into chunks on its section heading
    marker ("### "). No structured facts here - this is prose guidance, not spec/price data."""
    text = path.read_text(encoding="utf-8").strip()
    parts = re.split(rf"\n(?={re.escape(heading_prefix)})", text)
    return [(part.strip(), None) for part in parts if len(part.strip()) >= _MIN_CHUNK_LENGTH]


def _load_all_chunks() -> tuple[list[str], list[dict[str, str] | None]]:
    chunks: list[str] = []
    structured: list[dict[str, str] | None] = []
    if not KB_DIR.is_dir():
        logger.error("knowledge base directory not found: %s", KB_DIR)
        return chunks, structured

    for path in sorted(KB_DIR.iterdir()):
        try:
            if path.suffix == ".html":
                pairs = _chunk_html(path)
            elif path.name == "maruti_prices_kb_tts.md":
                pairs = _chunk_price_markdown(path)
            elif path.suffix == ".md":
                pairs = _chunk_delimited_text(path, "## ")
            elif path.suffix == ".txt":
                pairs = _chunk_delimited_text(path, "### ")
            else:
                continue
        except Exception:
            logger.error("failed to parse knowledge base file %s", path, exc_info=True)
            continue
        for chunk_text, fact in pairs:
            chunks.append(chunk_text)
            structured.append(fact)
    return chunks, structured


# ---------------------------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------------------------
# Embeddings (semantic retrieval)
# ---------------------------------------------------------------------------------------------


def _corpus_fingerprint(chunks: list[str]) -> str:
    """Content hash of the corpus, used to invalidate the cached embeddings if the KB changes."""
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(chunk.encode("utf-8"))
    return hasher.hexdigest()


def _load_embedding_model():
    """Construct the local ONNX embedding model. Returns None (not raises) on any failure, so
    a missing/broken embedding model degrades to BM25-only retrieval instead of crashing the
    whole knowledge base."""
    try:
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=EMBEDDING_MODEL_NAME, cache_dir=str(_MODEL_CACHE_DIR))
    except Exception:
        logger.error(
            "failed to load embedding model %s, falling back to BM25-only retrieval",
            EMBEDDING_MODEL_NAME,
            exc_info=True,
        )
        return None


def _load_or_compute_chunk_embeddings(chunks: list[str], model) -> "np.ndarray | None":
    """Load cached (L2-normalized) chunk embeddings if the corpus hasn't changed since they
    were computed, else compute and cache them. Computing embeddings for the full corpus takes
    tens of seconds on CPU - paying that once (and caching across process restarts, and baking
    the result into the Docker image at build time) keeps it off the per-call and per-process
    hot path."""
    if model is None or not chunks:
        return None

    fingerprint = _corpus_fingerprint(chunks)
    if _EMBEDDINGS_CACHE_FILE.exists():
        try:
            cached = np.load(_EMBEDDINGS_CACHE_FILE, allow_pickle=False)
            if cached["fingerprint"].item() == fingerprint and len(cached["vectors"]) == len(
                chunks
            ):
                return cached["vectors"]
        except Exception:
            logger.warning("failed to read cached KB embeddings, recomputing", exc_info=True)

    logger.info("computing embeddings for %d KB chunks (cache miss)...", len(chunks))
    vectors = np.array(list(model.embed(chunks, batch_size=64)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms > 0, norms, 1.0)

    try:
        _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        np.savez(_EMBEDDINGS_CACHE_FILE, vectors=vectors, fingerprint=fingerprint)
    except Exception:
        logger.warning("failed to write KB embeddings cache", exc_info=True)

    return vectors


# Cosine similarity is NOT a reliable "nothing relevant" signal on its own - measured directly:
# a nonsense query ("zzqxw plonk fnorble") scored a top similarity of 0.525, HIGHER than a
# genuine query ("Brezza ki keemat kya hai", 0.431). Sentence embeddings don't have a
# well-calibrated zero-point the way BM25 does (zero shared vocabulary = zero score), so
# embeddings are only trusted to independently vouch for "there is relevant content" above this
# much more conservative bar; below it, they still freely help rank/expand recall once BM25 has
# already established some lexical basis for the query (see search()).
_MIN_STANDALONE_EMBEDDING_SIMILARITY = 0.65


def _embedding_rank(query: str, min_similarity: float = 0.0) -> list[int]:
    """Rank chunk indices by cosine similarity to `query`'s embedding, best first, keeping only
    those at or above `min_similarity`. Empty list if the embedding model/vectors aren't
    available (BM25-only fallback)."""
    if _EMBEDDING_MODEL is None or _EMBEDDINGS is None:
        return []
    query_vec = next(iter(_EMBEDDING_MODEL.embed([query])))
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm
    similarities = _EMBEDDINGS @ query_vec
    ranked = sorted(range(len(_CHUNKS)), key=lambda i: similarities[i], reverse=True)
    return [i for i in ranked if similarities[i] >= min_similarity]


# ---------------------------------------------------------------------------------------------
# Fusion and re-ranking
# ---------------------------------------------------------------------------------------------


def _reciprocal_rank_fusion(rankings: list[list[int]], k: int = _RRF_K) -> list[int]:
    """Combine multiple best-first index rankings into one fused ranking: score(i) = sum over
    rankings containing i of 1/(k + rank). Simple, and doesn't require the different
    retrievers' raw scores to be on comparable scales (BM25 scores and cosine similarities
    aren't)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


def _rerank(query: str, candidate_indices: list[int]) -> list[int]:
    """Lightweight re-rank over the fused candidates: boost chunks containing more of the
    query's exact words verbatim. Not a learned/cross-encoder re-rank - deliberately simple, no
    extra model or network call. Python's sort is stable, so candidates that tie on this signal
    keep their existing (fused) relative order rather than being shuffled arbitrarily."""
    query_words = set(_tokenize(query))
    if not query_words:
        return candidate_indices

    def overlap(idx: int) -> float:
        chunk_words = set(_tokenize(_CHUNKS[idx]))
        return len(query_words & chunk_words) / len(query_words)

    return sorted(candidate_indices, key=overlap, reverse=True)


# ---------------------------------------------------------------------------------------------
# Hindi/Hinglish query expansion
# ---------------------------------------------------------------------------------------------

# Explicit Hindi/Hinglish -> English map for common car-buying vocabulary. See the module
# docstring for why this exists alongside the embedding model rather than instead of it.
_HI_EN_SYNONYMS = {
    "keemat": "price",
    "kimat": "price",
    "daam": "price",
    "kharcha": "cost",
    "jaankari": "information",
    "suraksha": "safety",
    "surakshit": "safe",
    "rang": "colour",
    "milega": "available",
    "milta": "available",
    "uplabdh": "available",
    "dikhao": "show",
}


def _expand_query(query: str) -> str:
    """Append English equivalents for any known Hindi/Hinglish term in `query`. Appends rather
    than replaces, so the embedding model still sees the original words too (it may carry some
    cross-lingual signal beyond this explicit map, just not reliably enough to depend on alone).
    """
    lowered = query.lower()
    extra = [english for hindi, english in _HI_EN_SYNONYMS.items() if hindi in lowered]
    return f"{query} {' '.join(extra)}" if extra else query


# ---------------------------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------------------------


def _format_structured_fact(fact: dict[str, str]) -> str:
    label = fact.get("variant") or fact.get("model") or "?"
    other = {k: v for k, v in fact.items() if k not in ("variant", "model") and v}
    details = " | ".join(f"{key}: {value}" for key, value in other.items())
    return f"{label} | {details}" if details else label


def _format_results(indices: list[int]) -> str:
    prose_lines: list[str] = []
    structured_lines: list[str] = []
    seen: set[tuple] = set()

    for idx in indices:
        prose_lines.append(_CHUNKS[idx])
        fact = _STRUCTURED[idx]
        if fact:
            key = tuple(sorted(fact.items()))
            if key not in seen:
                seen.add(key)
                structured_lines.append(_format_structured_fact(fact))

    sections = []
    if structured_lines:
        sections.append(
            "STRUCTURED FACTS (verbatim from KB, one variant/spec per line - never blend "
            "numbers across lines or across variants):\n"
            + "\n".join(f"- {line}" for line in structured_lines)
        )
    sections.append("PASSAGES:\n" + "\n\n---\n\n".join(prose_lines))
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------------------------
# Module-level load (once per worker process - see the module docstring)
# ---------------------------------------------------------------------------------------------

_CHUNKS, _STRUCTURED = _load_all_chunks()
_BM25 = BM25Okapi([_tokenize(chunk) for chunk in _CHUNKS]) if _CHUNKS else None
_EMBEDDING_MODEL = _load_embedding_model()
_EMBEDDINGS = _load_or_compute_chunk_embeddings(_CHUNKS, _EMBEDDING_MODEL)

logger.info(
    "knowledge base loaded: %d chunks from %s (embeddings: %s)",
    len(_CHUNKS),
    KB_DIR,
    "enabled" if _EMBEDDINGS is not None else "disabled",
)


def search(query: str, top_k: int = 3, rerank: bool = True) -> str:
    """Return the top_k most relevant KB chunks for `query` (BM25 + semantic embeddings fused
    via RRF, optionally re-ranked), or an honest "nothing found" message the agent can pass
    along per Hardcoded Rule One. `rerank` is exposed for debugging/evaluation; leave it on in
    production.
    """
    if _BM25 is None:
        return "The knowledge base is empty or failed to load."

    expanded_query = _expand_query(query)
    tokens = _tokenize(expanded_query)
    if not tokens:
        return "No relevant information found in the knowledge base for this query."

    bm25_scores = _BM25.get_scores(tokens)
    bm25_ranking = [
        i
        for i in sorted(range(len(_CHUNKS)), key=lambda i: bm25_scores[i], reverse=True)
        if bm25_scores[i] > 0
    ]
    # BM25 finding nothing means zero shared vocabulary with the entire corpus (even after
    # Hindi/Hinglish synonym expansion) - a strong "probably off-topic" signal. In that case,
    # only let the embedding ranking in if it clears a high confidence bar on its own (see
    # _MIN_STANDALONE_EMBEDDING_SIMILARITY); otherwise trust it freely to help rank/expand
    # recall alongside BM25's already-established lexical match.
    min_similarity = 0.0 if bm25_ranking else _MIN_STANDALONE_EMBEDDING_SIMILARITY
    embedding_ranking = _embedding_rank(expanded_query, min_similarity=min_similarity)

    fused = _reciprocal_rank_fusion([ranking for ranking in (bm25_ranking, embedding_ranking) if ranking])
    if not fused:
        return "No relevant information found in the knowledge base for this query."

    candidate_pool = fused[: max(top_k * 4, 10)]
    top_indices = _rerank(expanded_query, candidate_pool)[:top_k] if rerank else candidate_pool[:top_k]

    return _format_results(top_indices)
