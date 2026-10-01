"""Authentication and role-based authorization for the FleetMind API.

Split into three small modules so each is testable on its own:

    passwords.py  bcrypt hashing and constant-time verification
    tokens.py     JWT issue/decode (HS256)
    deps.py       FastAPI dependencies: who is calling, and may they

The enforcement rule this package exists to make true: **every restriction is
applied at the API layer**. Hiding a button in the frontend is a usability
choice, never a security one, so each scoped endpoint re-derives what the
caller is allowed to see from the token, not from a query parameter the
client supplied.
"""
