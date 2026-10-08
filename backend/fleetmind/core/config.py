"""Application configuration via environment variables (12-factor)."""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
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

    # LLM narrative layer (fleetmind/agents/llm.py). Absent by default and read
    # from backend/.env like every other setting here, so one file configures
    # the whole backend. With no key the rule-based decision layer still works
    # in full; only narrative generation returns 503.
    #
    # ANTHROPIC_API_KEY is accepted as an alias purely for continuity: the key
    # was first added to .env under that name while the layer still targeted
    # Claude. The provider is Groq now (see agents/llm.py), so GROQ_API_KEY is
    # the correct name and the alias can be dropped once .env is updated.
    groq_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GROQ_API_KEY", "ANTHROPIC_API_KEY"),
    )
    groq_model: str = "openai/gpt-oss-120b"

    # Auth. jwt_secret has NO fallback on purpose -- unlike database_url
    # above, a well-known default signing key would let anyone who has read
    # this repository mint an admin token. When it is unset, auth/tokens.py
    # generates a random per-process secret and logs a warning: tokens stop
    # surviving a restart, which is a visible nuisance rather than a silent
    # hole. Set JWT_SECRET in backend/.env for anything beyond a local demo.
    jwt_secret: str | None = None
    access_token_ttl_minutes: int = 720  # 12h: long enough for a demo session

    # Which path makes maintenance decisions (OPEN_QUESTIONS.md Q6).
    #   "expected_cost" -- the calibrated-decision layer: the action with least
    #                      expected cost (decision/policy.py), and a machine is
    #                      flagged by a repeated SPRT over its windows
    #                      (decision/sequential.py). The authoritative path.
    #   "heuristic"     -- the original four-agent chain in agents/agents.py,
    #                      flagging on ANY HIGH/CRITICAL window. Kept so every
    #                      documented heuristic figure stays reproducible.
    decision_policy: Literal["expected_cost", "heuristic"] = "expected_cost"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
