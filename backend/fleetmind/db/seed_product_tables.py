"""One-time seed for the product-layer tables (customers, technicians, equipment).

Also assigns each equipment row a linked_experiment_id: a deterministic
round-robin over its depot's experiments, ordered by id. CWRU has no real
per-motor identity, so this link is simulated demo data, not measured
telemetry — see db/models.py:Equipment.linked_experiment_id.

Usage:
    python -m fleetmind.db.seed_product_tables
"""

from __future__ import annotations

from sqlalchemy import insert, select

from fleetmind.db.models import (
    Customer,
    Equipment,
    Experiment,
    Technician,
    seed_customers_payload,
    seed_equipment_payload,
    seed_technicians_payload,
)
from fleetmind.db.session import create_db_engine, make_session_factory


def main() -> None:
    engine = create_db_engine()
    factory = make_session_factory(engine)
    session = factory()

    if session.execute(select(Customer.id)).first() is None:
        session.execute(insert(Customer), seed_customers_payload())
        session.flush()
        print(f"  seeded {len(seed_customers_payload())} customers")

    if session.execute(select(Technician.id)).first() is None:
        session.execute(insert(Technician), seed_technicians_payload())
        session.flush()
        print(f"  seeded {len(seed_technicians_payload())} technicians")

    if session.execute(select(Equipment.id)).first() is not None:
        print("Equipment already seeded — skipping. Delete rows first to reseed.")
        session.close()
        return

    experiments_by_depot: dict[int, list[int]] = {}
    equipment_payload = seed_equipment_payload()
    depot_position: dict[int, int] = {}

    for row in equipment_payload:
        depot_id = row["depot_id"]
        if depot_id not in experiments_by_depot:
            experiments_by_depot[depot_id] = [
                exp_id
                for exp_id, in session.execute(
                    select(Experiment.id)
                    .where(Experiment.depot_id == depot_id)
                    .order_by(Experiment.id)
                ).all()
            ]
        exps = experiments_by_depot[depot_id]
        i = depot_position.get(depot_id, 0)
        row["linked_experiment_id"] = exps[i % len(exps)]
        depot_position[depot_id] = i + 1

    session.execute(insert(Equipment), equipment_payload)
    session.flush()
    session.commit()
    print(f"  seeded {len(equipment_payload)} equipment rows with linked_experiment_id")
    session.close()


if __name__ == "__main__":
    main()
