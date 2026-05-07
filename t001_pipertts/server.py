import io
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse

import numpy as np
import soundfile as sf
import wave

from .t001_settings import Settings
from .piper_config import SynthesisConfig

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
syn_config = SynthesisConfig(
    speaker_id=1,
    volume=0.5,  # half as loud
    length_scale=1.5,  # twice as slow
    noise_scale=0.0,  # more audio variation
    noise_w_scale=0.0,  # more speaking variation
    normalize_audio=False, # use raw audio from voice
)

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

@app.get("/stream-audio")
async def stream_audio(
    text: str = Query(..., description="Văn bản cần chuyển sang tiếng nói"),
):
    if "piper_voice" not in models:
        raise HTTPException(503, "Piper TTS model is not available")

    voice = models["piper_voice"]
    sample_rate = 22050

    t0 = time.perf_counter()
        
    try:
        audio = voice.synthesize(text, syn_config=syn_config)
        
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            first_chunk = True

            for chunk in audio:
                if first_chunk:
                    wav_file.setnchannels(chunk.sample_channels)
                    wav_file.setsampwidth(chunk.sample_width)
                    wav_file.setframerate(chunk.sample_rate)
                    first_chunk = False

                wav_file.writeframes(chunk.audio_int16_bytes)
        
        buf.seek(0)
        raw_wav = buf.read()
        print(f"{text} -> {len(raw_wav)}")

        elapsed = time.perf_counter() - t0
        num_samples = (len(raw_wav) - 44) / 2
        duration = num_samples / sample_rate 
        
        headers = {
            "X-Engine": "piper",
            "X-Duration-Seconds": f"{duration:.3f}",
            "X-Inference-Time": f"{elapsed:.3f}",
            "X-Sample-Rate": str(sample_rate),
            "Content-Disposition": f'attachment; filename="piper_tts.wav"',
        }
        return streaming_audio_response(raw_wav, "wav", headers)

    except Exception as e:
        logger.exception("Piper synthesis failed")
        raise HTTPException(500, f"Synthesis error: {e}")

# ════════════════════════════════════════════════════════════════════════════
# Run
# ════════════════════════════════════════════════════════════════════════════ 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=11141, reload=False)