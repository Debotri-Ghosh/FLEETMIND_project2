"""Create the documented demo logins. Safe to re-run.

    python -m fleetmind.db.seed_users              # create missing users
    python -m fleetmind.db.seed_users --reset-passwords

Run this *after* `seed_product_tables.py`, because customer and technician
rows have to exist for the role scopes to point at anything.

The passwords are weak and published in the README on purpose: a reviewer has
to be able to log in as each role and check that the restrictions are real.
Two operators sit at different sites so the site scoping is demonstrable
rather than theoretical.
"""

from __future__ import annotations

import argparse

from sqlalchemy import select

from fleetmind.auth.passwords import hash_password
from fleetmind.core.config import get_settings
from fleetmind.db.models import Customer, Depot, Technician, User, seed_users_payload
from fleetmind.db.session import create_db_engine, make_session_factory


def _resolve_scope(session, spec: dict) -> dict:
    """Turn the human-readable names in the payload into real foreign keys.

    Names rather than hardcoded ids, because the seed order is not guaranteed
    to produce the same ids on a fresh database -- and a login silently bound
    to the wrong site would make the whole access-control demo meaningless.
    """
    values: dict[str, object] = {}

    if depot_name := spec.get("depot_name"):
        depot = session.execute(select(Depot).where(Depot.name == depot_name)).scalar_one_or_none()
        if depot is None:
            raise SystemExit(f"site {depot_name!r} not found -- seed depots first")
        values["depot_id"] = depot.id

    if customer_name := spec.get("customer_name"):
        customer = session.execute(
            select(Customer).where(Customer.name == customer_name)
        ).scalar_one_or_none()
        if customer is None:
            raise SystemExit(
                f"customer {customer_name!r} not found -- run seed_product_tables.py first"
            )
        values["customer_id"] = customer.id

    if technician_name := spec.get("technician_name"):
        technician = session.execute(
            select(Technician).where(Technician.name == technician_name)
        ).scalar_one_or_none()
        # Optional link: an operator login works without a technician row.
        if technician is not None:
            values["technician_id"] = technician.id

    return values


def seed_users(session, reset_passwords: bool = False) -> dict[str, int]:
    created = updated = skipped = 0

    for spec in seed_users_payload():
        email = spec["email"]
        existing = session.execute(select(User).where(User.email == email)).scalar_one_or_none()

        if existing is not None:
            if reset_passwords:
                existing.hashed_password = hash_password(spec["password"])
                updated += 1
            else:
                skipped += 1
            continue

        session.add(
            User(
                email=email,
                hashed_password=hash_password(spec["password"]),
                full_name=spec.get("full_name"),
                role=spec["role"],
                **_resolve_scope(session, spec),
            )
        )
        created += 1

    session.commit()
    return {"created": created, "password_reset": updated, "unchanged": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed FleetMind demo logins.")
    parser.add_argument(
        "--reset-passwords",
        action="store_true",
        help="Re-hash the documented password for users that already exist.",
    )
    args = parser.parse_args()

    session = make_session_factory(create_db_engine(get_settings().database_url))()
    result = seed_users(session, reset_passwords=args.reset_passwords)
    session.close()

    print(
        f"{result['created']} users created, "
        f"{result['password_reset']} passwords reset, "
        f"{result['unchanged']} left unchanged."
    )
    print("Demo credentials are listed in README.md -- they are not secrets.")


if __name__ == "__main__":
    main()
