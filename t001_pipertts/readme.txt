I. Log work
1. Cài đặt piper-tts
    - onnxruntime
    - piper-tts
2. Cài đặt fastapi

II. Sử dụng
1. Cách download model piper_tts:
    Mạng tìm: piper việt nam
    Một số link:
        https://github.com/phatjkk/vits-tts-vietnamese
            download model trong thư mục fine-tuning-model

2. Đặt config trong
    t001_pipertts/t001_settings.py hoặc .env

3. Cách chạy
    Ở Terminal:
        uvicorn t001_pipertts.server:app --reload --port 11140
    Hoặc vào debug (Ctrl + Shift + B)
        