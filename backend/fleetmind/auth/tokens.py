"""JWT issue and decode (HS256).

PyJWT rather than hand-rolled HMAC: algorithm-confusion and signature-timing
bugs are easy to write and hard to notice, and `jwt.decode` with an explicit
`algorithms=` list is the documented way to avoid the first of them.

The token carries only the user id, the role and -- for operators -- the site
they are scoped to. Everything else is looked up from the database on each
request, so revoking or re-scoping a user takes effect on their next call
rather than when their token happens to expire.
"""

from __future__ import annotations

import logging
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from fleetmind.core.config import get_settings

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"

# Generated once per process when no secret is configured. This deliberately
# fails *safe* rather than shipping a well-known default key: tokens stop
# working across a restart, which is a visible nuisance, instead of silently
# being forgeable by anyone who has read this repository.
_EPHEMERAL_SECRET = secrets.token_urlsafe(48)
_warned = False


class InvalidToken(Exception):
    """The token is missing, malformed, expired or wrongly signed."""


def get_secret() -> str:
    global _warned
    configured = get_settings().jwt_secret
    if configured:
        return configured
    if not _warned:
        logger.warning(
            "JWT_SECRET is not set; using a random per-process secret. Every "
            "restart will invalidate all issued tokens. Set JWT_SECRET in "
            "backend/.env for anything beyond a local demo."
        )
        _warned = True
    return _EPHEMERAL_SECRET


def create_access_token(
    user_id: int,
    role: str,
    depot_id: int | None = None,
    customer_id: int | None = None,
    expires_in: timedelta | None = None,
) -> str:
    settings = get_settings()
    ttl = expires_in or timedelta(minutes=settings.access_token_ttl_minutes)
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "role": role,
        "depot_id": depot_id,
        "customer_id": customer_id,
        "iat": now,
        "exp": now + ttl,
    }
    return jwt.encode(payload, get_secret(), algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """Verify and decode, or raise InvalidToken.

    `algorithms=[ALGORITHM]` is not optional: without it a token claiming
    `alg: none` would be accepted.
    """
    try:
        payload = jwt.decode(token, get_secret(), algorithms=[ALGORITHM])
    except jwt.PyJWTError as exc:
        raise InvalidToken(str(exc)) from exc

    if "sub" not in payload or "role" not in payload:
        raise InvalidToken("token is missing required claims")
    return payload
