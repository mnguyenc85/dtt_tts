from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MODEL_PATH: str = 'D:/pythons/_models/piper_tts/'
    # MODEL_NAME: str = 'vi_VN-vais1000-medium'
    MODEL_NAME: str = 'phatjkk_finetuning_v2'

    # Cấu hình để đọc từ file .env
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore")

settings = Settings()