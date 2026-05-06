import io
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

import numpy as np
import soundfile as sf

from .t001_settings import Settings

# region Helpers ────────────────────────────────────────────────────────────

MIME = {"wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg"} 
 
def audio_to_bytes(audio: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    """Convert numpy audio array → bytes in the requested format."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format=fmt.upper())
    buf.seek(0)
    return buf.read()
 
def streaming_audio_response(audio_bytes: bytes, fmt: str, headers: dict) -> StreamingResponse:
    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type=MIME[fmt],
        headers=headers,
    )
# endregion

# ════════════════════════════════════════════════════════════════════════════
# Main program
# ════════════════════════════════════════════════════════════════════════════ 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Global model holders ───────────────────────────────────────────────────

settings = Settings()
models: dict = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
    logger.info("Loading TTS models...")

    # ── Piper TTS ──
    try:
        from piper.voice import PiperVoice
 
        piper_model_path = f"{settings.MODEL_PATH}{settings.MODEL_NAME}.onnx"
        logger.info(f"Loading Piper from {piper_model_path} …")
        models["piper_voice"] = PiperVoice.load(piper_model_path)
        logger.info("✓ Piper TTS ready")
    except Exception as e:
        logger.warning(f"Piper TTS not loaded: {e}")
 
    yield  # ── app runs here ──
 
    logger.info("Shutting down, releasing models …")
    models.clear()


app = FastAPI(
    title="TTS API",
    description="Text-to-Speech API powered by Piper TTS",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── Routers ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "engines": {
            "piper": "piper_voice" in models,
        },
    }

# ════════════════════════════════════════════════════════════════════════════
# Run
# ════════════════════════════════════════════════════════════════════════════ 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=11140, reload=False)