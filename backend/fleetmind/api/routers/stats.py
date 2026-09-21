"""Aggregate statistics endpoints."""

import json
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from fleetmind.api.schemas import (
    DepotModelMetrics,
    FaultDistributionEntry,
    FederatedSummary,
    ModelPerformance,
    ProductSummary,
)
from fleetmind.db import session as db_session
from fleetmind.db.models import Depot, Equipment, ServiceRequest, SignalWindow, WorkOrder

router = APIRouter(prefix="/stats", tags=["stats"])

MLRUNS = Path("mlruns")
_OPEN_REQUEST_STATUSES = ("reported", "diagnosed", "dispatched", "in_progress")
_OPEN_WORK_ORDER_STATUSES = ("assigned", "traveling", "in_progress")


@router.get("/fault-distribution", response_model=list[FaultDistributionEntry])
def fault_distribution(db: Session = Depends(db_session.get_session)):
    rows = db.execute(
        select(SignalWindow.label, func.count().label("n_windows"))
        .group_by(SignalWindow.label)
        .order_by(func.count().desc())
    )
    return [FaultDistributionEntry(label=label, n_windows=n) for label, n in rows]


@router.get("/model-performance", response_model=ModelPerformance)
def model_performance(db: Session = Depends(db_session.get_session)):
    """Real numbers from mlruns/ — see CLAUDE.md's "Real, current results" section.
    Not regenerated here; this just reads what train.py/federated/run.py already wrote."""
    depot_names = dict(db.execute(select(Depot.id, Depot.name)).all())

    per_depot = []
    for depot_id in sorted(depot_names):
        metrics_path = MLRUNS / f"depot_{depot_id}" / "metrics.json"
        if not metrics_path.exists():
            continue
        m = json.loads(metrics_path.read_text())
        per_depot.append(
            DepotModelMetrics(
                depot_id=depot_id,
                depot_name=depot_names[depot_id],
                best_val_acc=m["best_val_acc"],
                train_windows=m["train_windows"],
                val_windows=m["val_windows"],
            )
        )

    federated = None
    federated_path = MLRUNS / "federated" / "fedprox" / "federated.json"
    if federated_path.exists():
        f = json.loads(federated_path.read_text())
        federated = FederatedSummary(
            strategy=f["strategy"],
            rounds=f["rounds"],
            mean_baseline_accuracy=f["mean_baseline_accuracy"],
            mean_final_accuracy=f["mean_final_accuracy"],
            gain_vs_baseline=f["gain_vs_baseline"],
            per_depot_final_accuracy={
                int(depot_id): row["accuracy"] for depot_id, row in f["final"].items()
            },
        )

    return ModelPerformance(per_depot=per_depot, federated=federated)


@router.get("/product-summary", response_model=ProductSummary)
def product_summary(db: Session = Depends(db_session.get_session)):
    equipment_by_status = dict(
        db.execute(
            select(Equipment.status, func.count()).group_by(Equipment.status)
        ).all()
    )

    open_requests_by_priority = dict(
        db.execute(
            select(ServiceRequest.priority, func.count())
            .where(ServiceRequest.status.in_(_OPEN_REQUEST_STATUSES))
            .group_by(ServiceRequest.priority)
        ).all()
    )

    ai_flagged_open = db.execute(
        select(func.count()).select_from(ServiceRequest).where(
            ServiceRequest.status.in_(_OPEN_REQUEST_STATUSES),
            ServiceRequest.agent_decision_id.is_not(None),
        )
    ).scalar_one()

    active_work_orders = db.execute(
        select(func.count())
        .select_from(WorkOrder)
        .where(WorkOrder.status.in_(_OPEN_WORK_ORDER_STATUSES))
    ).scalar_one()

    return ProductSummary(
        equipment_by_status=equipment_by_status,
        open_requests_by_priority=open_requests_by_priority,
        ai_flagged_open=ai_flagged_open,
        active_work_orders=active_work_orders,
    )
