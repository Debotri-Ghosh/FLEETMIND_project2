"""Login and identity endpoints.

`POST /auth/login` is the only unauthenticated route in the product API
besides the liveness probe -- everything else requires the token it issues.
"""

from __future__ import annotations

import logging
import secrets
from typing import cast

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from fleetmind.api.schemas import CurrentUser, LoginRequest, Role, TokenResponse
from fleetmind.auth.deps import get_current_user
from fleetmind.auth.passwords import hash_password, verify_password
from fleetmind.auth.tokens import create_access_token
from fleetmind.core.config import get_settings
from fleetmind.db import session as db_session
from fleetmind.db.models import Customer, Depot, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# A real bcrypt hash of a random secret, computed once at import. It must be
# genuinely valid: verify_password() returns early on a malformed hash, which
# would make the unknown-email path measurably faster than the wrong-password
# path and reintroduce exactly the enumeration oracle this is here to close.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(32))


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(db_session.get_session)):
    """Exchange email + password for a bearer token.

    A wrong password and an unknown email produce the identical 401, and the
    password is verified even when no user was found, so response timing does
    not reveal which accounts exist.
    """
    user = db.execute(
        select(User).where(User.email == payload.email.lower().strip())
    ).scalar_one_or_none()

    # A fixed bcrypt hash of a value no one can supply, used only to burn the
    # same CPU time on the unknown-email path as on the wrong-password path.
    reference = user.hashed_password if user else _DUMMY_HASH
    password_ok = verify_password(payload.password, reference)

    if user is None or not password_ok or not user.is_active:
        logger.info("failed login attempt for %r", payload.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(
        user_id=user.id,
        role=user.role,
        depot_id=user.depot_id,
        customer_id=user.customer_id,
    )
    return TokenResponse(
        access_token=token,
        expires_in_minutes=get_settings().access_token_ttl_minutes,
        user=_current_user(db, user),
    )


@router.get("/me", response_model=CurrentUser)
def me(
    user: User = Depends(get_current_user),
    db: Session = Depends(db_session.get_session),
):
    """Who the caller is, and what they are scoped to.

    The frontend reads this on load to decide what to render. That is a
    usability decision only -- every restriction it reflects is independently
    enforced on the endpoints themselves.
    """
    return _current_user(db, user)


def _current_user(db: Session, user: User) -> CurrentUser:
    site_name = None
    if user.depot_id is not None:
        depot = db.get(Depot, user.depot_id)
        site_name = depot.name if depot else None
    customer_name = None
    if user.customer_id is not None:
        customer = db.get(Customer, user.customer_id)
        customer_name = customer.name if customer else None
    return CurrentUser(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        # The ck_users_role CHECK constraint guarantees this is one of the
        # three literals; mypy cannot see a database constraint.
        role=cast(Role, user.role),
        customer_id=user.customer_id,
        customer_name=customer_name,
        depot_id=user.depot_id,
        site_name=site_name,
        technician_id=user.technician_id,
    )
