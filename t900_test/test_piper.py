# Thử piper-tts
# Models: 
#   https://github.com/phatjkk/vits-tts-vietnamese.git
#   https://huggingface.co/rhasspy/piper-voices/tree/3d796cc2f2c884b3517c527507e084f7bb245aea/vi/vi_VN/vais1000/medium

import wave
from piper import PiperVoice
from t001_pipertts.piper_config import SynthesisConfig

model_path = 'D:/pythons/_models/piper_tts/vi_VN-vais1000-medium/'

syn_config = SynthesisConfig(
    speaker_id=1,
    volume=0.2,  # half as loud
    length_scale=1.2,  # twice as slow
    noise_scale=1.0,  # more audio variation
    noise_w_scale=1.0,  # more speaking variation
    normalize_audio=False, # use raw audio from voice
)
voice = PiperVoice.load(f"{model_path}model.onnx")

audio = voice.synthesize("Hướng dẫn hiệu chuẩn đầu cân, T D A - 08B. Các bước thực hiện lần lượt là:", syn_config=syn_config)

with wave.open("data/out.wav", "wb") as wav_file:
    first_chunk = True

    for chunk in audio:
        if first_chunk:
            wav_file.setnchannels(chunk.sample_channels)
            wav_file.setsampwidth(chunk.sample_width)
            wav_file.setframerate(chunk.sample_rate)
            first_chunk = False

        wav_file.writeframes(chunk.audio_int16_bytes)