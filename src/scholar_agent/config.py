"""App settings, loaded from environment variables / the .env file."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: SecretStr
    # Used by the planner and researchers (many calls -> a cheap model makes sense).
    llm_model: str = "claude-haiku-5-5"
    # Used by the writer and fact-checker. Falls back to llm_model if unset.
    writer_model: str | None = "claude-sonnet-5-5"
    # How hard the model thinks: low | medium | high | xhigh | max.
    # Leave unset for Haiku 4.5, which doesn't support effort.
    llm_effort: str | None = None
    max_search_results: int = 5
    # Max drafts the writer may produce before the report is finalized anyway.
    max_drafts: int = 3
    # Where saved run state lives, so paused or crashed runs can be resumed.
    checkpoint_db: str = ".scholar/checkpoints.sqlite"


@lru_cache
def get_settings() -> Settings:
    return Settings()
