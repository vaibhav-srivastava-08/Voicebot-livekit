"""FastAPI server exposing the bridge's call endpoints and a local live-session debug page."""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from livekit import api

load_dotenv()

# Read directly from the environment (not agent.config) so the bridge stays importable on its
# own, matching Dockerfile.bridge, which never copies the agent/ package into its image.
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def debug_page() -> FileResponse:
    """Serve the local debug frontend for watching/joining a live agent session."""
    return FileResponse(STATIC_DIR / "debug.html")


@app.get("/token")
async def create_token(room: str = "debug-room", identity: str = "browser-user") -> dict:
    """Issue a short-lived LiveKit access token so the debug frontend can join a room."""
    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(
            api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True)
        )
        .to_jwt()
    )
    return {"token": token, "url": LIVEKIT_URL, "room": room}
