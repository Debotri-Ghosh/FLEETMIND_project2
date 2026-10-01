"""Application configuration via environment variables (12-factor)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # These fallbacks are intentional, not an oversight: this is a demo/thesis
    # project, not a production service, so a missing .env silently falling
    # back to a placeholder DB (which will simply fail to connect, surfacing
    # clearly at /health or on first query) is preferable to a hard startup
    # crash. It also lets pytest run with zero .env configuration at all, via
    # the sqlite override in tests/conftest.py. Do not "fix" this into a
    # fail-fast startup check without deciding that tradeoff on purpose.
    database_url: str = "postgresql://fleetmind:change_me@localhost:5432/fleetmind"
    cors_origins: str = "http://localhost:3000"
    log_level: str = "INFO"
    app_version: str = "0.1.0"

    # LLM narrative layer (fleetmind/agents/llm.py). Absent by default and
    # read from backend/.env like every other setting here, rather than from
    # the SDK's own ANTHROPIC_API_KEY lookup, so one file configures the whole
    # backend. With no key the rule-based decision layer still works in full;
    # only narrative generation returns 503.
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
