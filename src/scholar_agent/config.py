"""App settings, loaded from environment variables / the .env file."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: SecretStr
    llm_model: str = "claude-haiku-4-5"
    # How hard the model thinks: low | medium | high | xhigh | max
    # llm_effort: str = "low"
    max_search_results: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
