"""Password hashing.

bcrypt rather than a hand-rolled PBKDF2 loop: it is the standard choice, it
carries its own salt and cost factor inside the stored hash, and the library
installs as a prebuilt wheel (which matters on this locked-down laptop -- see
CLAUDE.md on the WSL2/Hyper-V restriction).
"""

from __future__ import annotations

import bcrypt

# bcrypt silently truncates input at 72 bytes, which would make two different
# long passwords equivalent. Reject rather than truncate: a password that
# "works" but is not the one the user set is worse than an error.
MAX_PASSWORD_BYTES = 72


class PasswordTooLong(ValueError):
    pass


def hash_password(password: str) -> str:
    raw = password.encode("utf-8")
    if len(raw) > MAX_PASSWORD_BYTES:
        raise PasswordTooLong(
            f"password must be at most {MAX_PASSWORD_BYTES} bytes (bcrypt truncates beyond that)"
        )
    return bcrypt.hashpw(raw, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Constant-time check. Returns False on a malformed hash rather than raising.

    A stored hash that bcrypt cannot parse means the row is corrupt, which is
    an authentication failure, not a 500 -- and raising here would also leak,
    via the error, that the account exists.
    """
    raw = password.encode("utf-8")
    if len(raw) > MAX_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(raw, hashed.encode("utf-8"))
    except ValueError:
        return False
