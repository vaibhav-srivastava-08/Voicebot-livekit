"""Tests for agent.knowledge_base: parsing the real MSIL KB files and hybrid search over them.

No mocking - parsing local KB files, BM25, and the local ONNX embedding model are all fast,
deterministic, offline operations (the embedding model and precomputed chunk embeddings are
cached to disk after the first run - see the module docstring - so this stays fast on repeat
test runs), so these tests run against the real "MSIL KNOWLEDGE BASE/" directory and the real
embedding model.
"""

from agent import knowledge_base


class TestLoading:
    def test_knowledge_base_directory_exists(self):
        assert knowledge_base.KB_DIR.is_dir(), (
            f"expected the KB directory at {knowledge_base.KB_DIR}; "
            "search() degrades to an empty-KB message without it"
        )

    def test_knowledge_base_loads_a_substantial_number_of_chunks(self):
        # The corpus has ~35 files covering model pages, Q&A articles, and price/variant lists -
        # a low count would mean a parser silently failed on most of them.
        assert len(knowledge_base._CHUNKS) > 50

    def test_structured_facts_extracted_for_a_substantial_share_of_chunks(self):
        # Price bullets and per-variant spec lines should yield structured facts; a near-zero
        # count would mean the extractors silently stopped matching the corpus's actual shape.
        structured_count = sum(1 for fact in knowledge_base._STRUCTURED if fact)
        assert structured_count > 100

    def test_embedding_model_loaded_successfully(self):
        # If this is None, the corpus's own files/environment are broken in a way that should
        # be visible in CI, not silently degrade every test to BM25-only.
        assert knowledge_base._EMBEDDING_MODEL is not None

    def test_chunk_embeddings_shape_matches_chunk_count(self):
        assert knowledge_base._EMBEDDINGS is not None
        assert knowledge_base._EMBEDDINGS.shape[0] == len(knowledge_base._CHUNKS)


class TestSearchInterfaceUnchanged:
    """search(query, top_k=3) must still work exactly as before - these are the pre-upgrade
    call patterns, unchanged."""

    def test_search_with_only_query_positional(self):
        result = knowledge_base.search("Brezza price")
        assert isinstance(result, str) and result

    def test_search_with_query_and_top_k_keyword(self):
        result = knowledge_base.search("Brezza price", top_k=2)
        assert isinstance(result, str) and result

    def test_search_for_a_known_model_returns_relevant_text(self):
        result = knowledge_base.search("Brezza price")
        assert "brezza" in result.lower()

    def test_search_for_safety_question_returns_relevant_text(self):
        result = knowledge_base.search("which car is safest airbags")
        assert "airbag" in result.lower()

    def test_search_for_nonsense_query_returns_honest_not_found_message(self):
        result = knowledge_base.search("zzqxw plonk fnorble asdkjaslkdj")
        assert "no relevant information" in result.lower()

    def test_search_for_empty_query_does_not_crash(self):
        result = knowledge_base.search("")
        assert isinstance(result, str)
        assert result != ""


class TestHindiHinglishCodeSwitching:
    """Queries using romanized Hindi/Hinglish car-buying terms that a keyword-only (BM25)
    search would fail on, since the KB corpus itself is entirely in English."""

    def test_search_handles_keemat_synonym_for_price(self):
        result = knowledge_base.search("Brezza ki keemat kya hai")
        lowered = result.lower()
        assert "brezza" in lowered
        assert "lakh" in lowered or "price" in lowered

    def test_search_handles_keemat_for_a_different_model(self):
        result = knowledge_base.search("What is the keemat of Alto K10")
        lowered = result.lower()
        assert "alto" in lowered
        assert "lakh" in lowered or "price" in lowered

    def test_search_handles_suraksha_synonym_for_safety(self):
        result = knowledge_base.search("Grand Vitara mein suraksha kaisi hai")
        lowered = result.lower()
        assert "safety" in lowered or "airbag" in lowered

    def test_expand_query_appends_english_synonym_for_known_hindi_term(self):
        expanded = knowledge_base._expand_query("Brezza ki keemat")
        assert "price" in expanded.lower()
        assert "keemat" in expanded.lower()  # original words kept, not replaced

    def test_expand_query_is_unchanged_when_no_hindi_synonym_present(self):
        query = "Brezza price"
        assert knowledge_base._expand_query(query) == query


class TestStructuredFacts:
    """For price/spec queries, structured {variant, price, specs} facts should be surfaced
    separately from prose, one fact per line, so numbers from different variants can't blend."""

    def test_split_on_last_colon_basic(self):
        assert knowledge_base._split_on_last_colon("Engine: 1.5 liter K15C") == (
            "Engine",
            "1.5 liter K15C",
        )

    def test_split_on_last_colon_uses_the_last_colon_for_variant_price_lines(self):
        text = "Brezza LXi MT, 1.5 liter, 5-speed MT, petrol: 8 lakh 11 thousand 400"
        assert knowledge_base._split_on_last_colon(text) == (
            "Brezza LXi MT, 1.5 liter, 5-speed MT, petrol",
            "8 lakh 11 thousand 400",
        )

    def test_split_on_last_colon_returns_none_without_a_colon(self):
        assert knowledge_base._split_on_last_colon("no colon here") is None

    def test_structured_from_variant_price_line_splits_variant_specs_and_price(self):
        text = "Brezza LXi MT, 1.5 liter, 5-speed MT, petrol (first variant): 8 lakh 11 thousand 400"
        fact = knowledge_base._structured_from_variant_price_line(text)
        assert fact == {
            "variant": "Brezza LXi MT",
            "price": "8 lakh 11 thousand 400",
            "specs": "1.5 liter, 5-speed MT, petrol (first variant)",
        }

    def test_structured_from_variant_feature_block_parses_multiple_labels(self):
        text = (
            "Variant: Brezza LXi CNG. Mileage: twenty five point five one km per kg. "
            "Key features: Bi-Halogen, six Airbags, Keyless Entry System."
        )
        fact = knowledge_base._structured_from_variant_feature_block(text)
        assert fact["variant"] == "Brezza LXi CNG"
        assert fact["mileage"] == "twenty five point five one km per kg"
        assert fact["key features"] == "Bi-Halogen, six Airbags, Keyless Entry System"

    def test_structured_from_variant_feature_block_none_without_variant_label(self):
        assert knowledge_base._structured_from_variant_feature_block("Mileage: ten km per liter") is None

    def test_search_for_price_query_includes_structured_facts_section(self):
        result = knowledge_base.search("Brezza LXi price")
        assert "STRUCTURED FACTS" in result
        assert "PASSAGES" in result
        assert result.index("STRUCTURED FACTS") < result.index("PASSAGES")

    def test_structured_facts_section_has_one_fact_per_line_not_merged(self):
        result = knowledge_base.search("Brezza variant prices", top_k=5)
        structured_block = result.split("PASSAGES")[0]
        fact_lines = [line for line in structured_block.splitlines() if line.startswith("- ")]
        # Each fact line should carry exactly one price value, never two (which would mean two
        # variants' numbers got merged onto the same line).
        for line in fact_lines:
            assert line.count("price:") <= 1


class TestFusionAndRerank:
    """Unit tests for the RRF fusion and lightweight re-rank helpers in isolation."""

    def test_reciprocal_rank_fusion_favors_items_ranked_well_by_multiple_retrievers(self):
        ranking_a = [10, 20, 30, 40]
        ranking_b = [20, 10, 40, 30]
        fused = knowledge_base._reciprocal_rank_fusion([ranking_a, ranking_b])
        # 10 and 20 are top-2 in both rankings; 30 and 40 are bottom-2 in both.
        assert set(fused[:2]) == {10, 20}
        assert set(fused[2:]) == {30, 40}

    def test_reciprocal_rank_fusion_handles_an_empty_ranking_list(self):
        assert knowledge_base._reciprocal_rank_fusion([]) == []

    def test_reciprocal_rank_fusion_includes_items_present_in_only_one_ranking(self):
        fused = knowledge_base._reciprocal_rank_fusion([[1, 2], [3]])
        assert set(fused) == {1, 2, 3}

    def test_rerank_boosts_chunk_with_more_exact_query_word_overlap(self, monkeypatch):
        monkeypatch.setattr(
            knowledge_base,
            "_CHUNKS",
            [
                "totally unrelated passage about something else entirely",
                "Brezza price and Brezza mileage details",
            ],
        )
        reranked = knowledge_base._rerank("Brezza price mileage", [0, 1])
        assert reranked == [1, 0]

    def test_rerank_is_stable_when_no_query_words_overlap(self, monkeypatch):
        monkeypatch.setattr(knowledge_base, "_CHUNKS", ["chunk a", "chunk b"])
        # Neither chunk shares words with the query, so overlap is 0 for both - order shouldn't
        # be shuffled (Python's sort is stable).
        assert knowledge_base._rerank("zzqxw nonsense", [0, 1]) == [0, 1]

    def test_search_still_works_with_rerank_disabled(self):
        result = knowledge_base.search("Brezza price", rerank=False)
        assert isinstance(result, str)
        assert "brezza" in result.lower()
