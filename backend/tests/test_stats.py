"""Tests for the dashboard stats endpoints: /stats/model-performance (reads
mlruns/ JSON, monkeypatched to a temp dir so tests don't depend on real
training artifacts existing) and /stats/product-summary (DB aggregates)."""

import json

from fleetmind.agents.sync_equipment_status import sync_equipment_status
from fleetmind.api.routers import stats as stats_module
from fleetmind.db.models import ServiceRequest


def test_model_performance_reads_mlruns(client, tmp_path, monkeypatch):
    depot_dir = tmp_path / "depot_1"
    depot_dir.mkdir()
    (depot_dir / "metrics.json").write_text(
        json.dumps({"depot_id": 1, "best_val_acc": 0.65, "train_windows": 100, "val_windows": 30})
    )
    fed_dir = tmp_path / "federated" / "fedprox"
    fed_dir.mkdir(parents=True)
    (fed_dir / "federated.json").write_text(
        json.dumps(
            {
                "strategy": "fedprox",
                "rounds": 30,
                "mean_baseline_accuracy": 0.6,
                "mean_final_accuracy": 0.43,
                "gain_vs_baseline": -0.17,
                "final": {"1": {"accuracy": 0.43}, "2": {"accuracy": 0.42}},
            }
        )
    )
    monkeypatch.setattr(stats_module, "MLRUNS", tmp_path)

    resp = client.get("/stats/model-performance")
    assert resp.status_code == 200
    body = resp.json()
    assert body["per_depot"] == [
        {
            "depot_id": 1,
            "depot_name": "depot_0hp",
            "best_val_acc": 0.65,
            "train_windows": 100,
            "val_windows": 30,
        }
    ]
    assert body["federated"]["strategy"] == "fedprox"
    assert body["federated"]["gain_vs_baseline"] == -0.17
    assert body["federated"]["per_depot_final_accuracy"] == {"1": 0.43, "2": 0.42}


def test_model_performance_missing_mlruns_files_is_graceful(client, tmp_path, monkeypatch):
    monkeypatch.setattr(stats_module, "MLRUNS", tmp_path)

    resp = client.get("/stats/model-performance")
    assert resp.status_code == 200
    body = resp.json()
    assert body["per_depot"] == []
    assert body["federated"] is None


def test_model_performance_skips_depots_without_metrics_file(client, tmp_path, monkeypatch):
    depot_dir = tmp_path / "depot_2"
    depot_dir.mkdir()
    (depot_dir / "metrics.json").write_text(
        json.dumps({"depot_id": 2, "best_val_acc": 0.72, "train_windows": 200, "val_windows": 50})
    )
    monkeypatch.setattr(stats_module, "MLRUNS", tmp_path)

    resp = client.get("/stats/model-performance")
    body = resp.json()
    assert [d["depot_id"] for d in body["per_depot"]] == [2]


def test_product_summary_counts(product_client, product_session):
    sync_equipment_status(product_session)  # files 1 AI request for equipment 1 (priority=high)

    product_client.post(
        "/service-requests",
        json={"customer_id": 1, "depot_id": 1, "title": "Urgent", "priority": "critical"},
    )
    resolved = product_client.post(
        "/service-requests",
        json={"customer_id": 1, "depot_id": 1, "title": "Old issue", "priority": "low"},
    ).json()
    resolved_row = product_session.get(ServiceRequest, resolved["id"])
    resolved_row.status = "resolved"
    product_session.commit()

    ai_requests = product_client.get("/service-requests", params={"ai_flagged": True}).json()
    product_client.post(
        "/work-orders",
        json={"service_request_id": ai_requests[0]["id"], "technician_id": 1},
    )

    resp = product_client.get("/stats/product-summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["equipment_by_status"] == {"active": 1, "maintenance": 1, "retired": 1}
    assert body["open_requests_by_priority"] == {"high": 1, "critical": 1}
    assert body["ai_flagged_open"] == 1
    assert body["active_work_orders"] == 1


def test_product_summary_empty_when_nothing_seeded(client):
    resp = client.get("/stats/product-summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["equipment_by_status"] == {}
    assert body["open_requests_by_priority"] == {}
    assert body["ai_flagged_open"] == 0
    assert body["active_work_orders"] == 0
