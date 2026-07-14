"""Agent class definition for the LiveKit voice assistant."""

from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.plugins import cartesia, deepgram, groq, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from agent.config import settings
from agent.prompts import HUMAN_TRANSFER_PHRASE, SYSTEM_PROMPT
from shared.logger import get_logger

logger = get_logger(__name__)


class Assistant(Agent):
    """LiveKit Agent subclass configured with the voicebot's persona and function tools."""

    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)

    @function_tool
    async def get_customer_info(self, ctx: RunContext, phone_number: str) -> dict:
        """Look up a customer's account details by phone number."""
        logger.info("get_customer_info called for phone_number=%s", phone_number)
        return {"name": "Test Customer", "account_status": "active"}

    @function_tool
    async def transfer_to_human(self, ctx: RunContext) -> str:
        """Transfer the current call to a human agent. Real transfer logic is wired in the bridge phase."""
        logger.info("transfer requested")
        return HUMAN_TRANSFER_PHRASE


def build_session() -> AgentSession:
    """Build an AgentSession wired with STT, LLM, TTS, VAD, and turn detection. Caller starts it."""
    return AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=groq.LLM(model="llama-3.3-70b-versatile", api_key=settings.groq_api_key),
        tts=cartesia.TTS(voice=settings.cartesia_voice_id),
        vad=silero.VAD.load(),
        turn_handling={
            "turn_detection": MultilingualModel(),
            # "adaptive" mode calls LiveKit Cloud's hosted interruption model, which is
            # unreachable from this self-hosted server and fails+retries on every turn.
            "interruption": {"mode": "vad"},
            # Preemptive generation speculatively starts the LLM/TTS before the user's turn is
            # confirmed done; on a slow local pipeline it can race with turn detection and stall.
            "preemptive_generation": {"enabled": False},
        },
    )
