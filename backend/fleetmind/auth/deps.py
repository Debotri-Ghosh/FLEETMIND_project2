"""FastAPI dependencies: who is calling, and what may they touch.

Two layers, deliberately separate:

  `get_current_user`  authenticates -- turns a bearer token into a live User
                      row, re-read from the database on every request so a
                      deactivated or re-scoped account takes effect at once.
  `require_roles`     authorizes -- refuses a role outright.

Scope (which *rows* a caller may see) is a third thing and is not a
dependency, because it is a filter rather than a yes/no: see `scope_*` below,
which each endpoint applies to its own query. The rule throughout is that the
caller's scope comes from their User row and never from a query parameter --
`?customer_id=2` must not be able to widen what a customer can read.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import InstrumentedAttribute, Session
from sqlalchemy.sql import ColumnElement

from fleetmind.auth.tokens import InvalidToken, decode_access_token
from fleetmind.db import session as db_session
from fleetmind.db.models import User

ROLE_CUSTOMER = "customer"
ROLE_OPERATOR = "operator"
ROLE_ADMIN = "admin"

# auto_error=False so a missing header produces our own 401 with a
# WWW-Authenticate challenge, rather than FastAPI's bare 403.
_bearer = HTTPBearer(auto_error=False)

_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(db_session.get_session),
) -> User:
    if credentials is None or not credentials.credentials:
        raise _UNAUTHENTICATED
    try:
        payload = decode_access_token(credentials.credentials)
    except InvalidToken as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    try:
        user_id = int(payload["sub"])
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="invalid token subject") from exc

    user = db.get(User, user_id)
    if user is None or not user.is_active:
        # Same message either way: whether an account exists is not something
        # an unauthenticated caller should be able to probe.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="account is not active",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # The token's role is a cached copy; the row is the truth. A role changed
    # in the database must not wait for the old token to expire.
    if payload.get("role") != user.role:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="token no longer matches this account; sign in again",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_roles(*roles: str):
    """Dependency factory: refuse any caller outside `roles`."""

    allowed = set(roles)

    def _check(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"role '{user.role}' may not access this resource "
                    f"(requires {' or '.join(sorted(allowed))})"
                ),
            )
        return user

    return _check


require_admin = require_roles(ROLE_ADMIN)
require_operator = require_roles(ROLE_OPERATOR, ROLE_ADMIN)
require_any_role = require_roles(ROLE_CUSTOMER, ROLE_OPERATOR, ROLE_ADMIN)


# ── row scoping ────────────────────────────────────────────────────────
#
# Returned as SQLAlchemy conditions rather than applied here, so each
# endpoint composes them into its own query and a reviewer can see exactly
# where the restriction lands.


def scope_by_customer(
    user: User, column: InstrumentedAttribute[int] | ColumnElement[int]
) -> ColumnElement[bool] | None:
    """Restrict to the caller's own customer, or None when unrestricted."""
    if user.role == ROLE_CUSTOMER:
        return column == user.customer_id
    return None


def scope_by_depot(
    user: User, column: InstrumentedAttribute[int] | ColumnElement[int]
) -> ColumnElement[bool] | None:
    """Restrict to the caller's own site, or None when unrestricted."""
    if user.role == ROLE_OPERATOR:
        return column == user.depot_id
    return None


def may_read_customer(user: User, customer_id: int | None) -> bool:
    if user.role == ROLE_ADMIN:
        return True
    if user.role == ROLE_CUSTOMER:
        return customer_id is not None and customer_id == user.customer_id
    return True  # operators triage on behalf of whoever owns the machine


def may_read_depot(user: User, depot_id: int | None) -> bool:
    if user.role == ROLE_OPERATOR:
        return depot_id is not None and depot_id == user.depot_id
    return True


def assert_may_read(user: User, *, customer_id: int | None, depot_id: int | None) -> None:
    """404, not 403, for a row outside the caller's scope.

    Returning 403 would confirm the row exists, which lets a customer
    enumerate other customers' equipment ids. "Not found" is what a caller
    outside the scope should observe, and it is also true from their side.
    """
    if not may_read_customer(user, customer_id) or not may_read_depot(user, depot_id):
        raise HTTPException(status_code=404, detail="not found")


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"
