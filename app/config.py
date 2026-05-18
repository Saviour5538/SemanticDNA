from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str
    spacy_model: str = "en_core_web_sm"
    gin_candidate_limit: int = 500
    entropy_smoothing: float = 0.5


settings = Settings()
