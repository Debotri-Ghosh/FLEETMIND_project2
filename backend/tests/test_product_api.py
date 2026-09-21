"""Coverage for the product-layer routers: equipment, customers, technicians,
service-requests, work-orders. See tests/conftest.py:product_client for the
seed data (1 customer, 1 technician, 3 equipment rows — one linked to an
experiment with a HIGH agent_decision, one unlinked, one retired)."""

from fleetmind.agents.sync_equipment_status import sync_equipment_status


def test_list_customers(product_client):
    resp = product_client.get("/customers")
    assert resp.status_code == 200
    customers = resp.json()
    assert len(customers) == 1
    assert customers[0]["name"] == "Acme Logistics"


def test_get_customer_404(product_client):
    assert product_client.get("/customers/999").status_code == 404


def test_list_technicians_filter_by_status(product_client):
    resp = product_client.get("/technicians", params={"status": "available"})
    assert resp.status_code == 200
    assert len(resp.json()) == 1

    resp = product_client.get("/technicians", params={"status": "busy"})
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_technician_404(product_client):
    assert product_client.get("/technicians/999").status_code == 404


def test_list_equipment(product_client):
    resp = product_client.get("/equipment")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 3
    assert rows[0]["linked_experiment_id"] == 1


def test_list_equipment_filters(product_client):
    resp = product_client.get("/equipment", params={"status": "retired"})
    assert resp.status_code == 200
    assert [r["name"] for r in resp.json()] == ["Retired Motor"]

    resp = product_client.get("/equipment", params={"customer_id": 1, "depot_id": 1})
    assert len(resp.json()) == 3

    resp = product_client.get("/equipment", params={"depot_id": 999})
    assert resp.json() == []


def test_get_equipment_404(product_client):
    assert product_client.get("/equipment/999").status_code == 404


def test_create_and_list_service_request(product_client):
    resp = product_client.post(
        "/service-requests",
        json={
            "customer_id": 1,
            "depot_id": 1,
            "equipment_id": 1,
            "title": "Unusual vibration noise",
            "description": "Loud clicking from the bearing housing",
            "priority": "high",
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "reported"
    assert body["priority"] == "high"
    request_id = body["id"]

    listed = product_client.get("/service-requests").json()
    assert len(listed) == 1
    assert listed[0]["id"] == request_id

    filtered = product_client.get("/service-requests", params={"equipment_id": 1}).json()
    assert len(filtered) == 1
    filtered_none = product_client.get("/service-requests", params={"equipment_id": 2}).json()
    assert filtered_none == []


def test_get_service_request_404(product_client):
    assert product_client.get("/service-requests/999").status_code == 404


def test_customer_filed_request_has_no_agent_context(product_client):
    resp = product_client.post(
        "/service-requests", json={"customer_id": 1, "depot_id": 1, "title": "Odd noise"}
    )
    body = resp.json()
    assert body["agent_decision_text"] is None
    assert body["agent_risk_score"] is None
    assert body["predicted_fault"] is None


def test_ai_flagged_service_request_carries_agent_context(product_client, product_session):
    sync_equipment_status(product_session)

    resp = product_client.get("/service-requests", params={"ai_flagged": True})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    row = rows[0]
    assert row["agent_decision_text"] == "[HIGH] elevated fault repetition rate"
    assert row["agent_risk_score"] == 0.8
    assert row["predicted_fault"] == "InnerRace"
    assert row["priority"] == "high"

    detail = product_client.get(f"/service-requests/{row['id']}").json()
    assert detail["agent_decision_text"] == "[HIGH] elevated fault repetition rate"


def test_ai_flagged_filter_excludes_customer_filed_requests(product_client, product_session):
    sync_equipment_status(product_session)
    product_client.post(
        "/service-requests", json={"customer_id": 1, "depot_id": 1, "title": "Manual report"}
    )

    ai_only = product_client.get("/service-requests", params={"ai_flagged": True}).json()
    assert len(ai_only) == 1
    assert ai_only[0]["title"].startswith("AI-flagged")

    customer_only = product_client.get("/service-requests", params={"ai_flagged": False}).json()
    assert len(customer_only) == 1
    assert customer_only[0]["title"] == "Manual report"


def test_create_service_request_defaults_priority_medium(product_client):
    resp = product_client.post(
        "/service-requests",
        json={"customer_id": 1, "depot_id": 1, "title": "Odd noise"},
    )
    assert resp.status_code == 201
    assert resp.json()["priority"] == "medium"


def test_work_order_lifecycle(product_client):
    sr = product_client.post(
        "/service-requests",
        json={"customer_id": 1, "depot_id": 1, "equipment_id": 1, "title": "Bearing noise"},
    ).json()
    assert sr["status"] == "reported"

    wo_resp = product_client.post(
        "/work-orders",
        json={"service_request_id": sr["id"], "technician_id": 1},
    )
    assert wo_resp.status_code == 201
    work_order = wo_resp.json()
    assert work_order["status"] == "assigned"

    # creating a work order dispatches the service request
    sr_after_dispatch = product_client.get(f"/service-requests/{sr['id']}").json()
    assert sr_after_dispatch["status"] == "dispatched"

    # advancing the work order propagates to the service request
    resp = product_client.patch(
        f"/work-orders/{work_order['id']}/status", json={"status": "in_progress"}
    )
    assert resp.status_code == 200
    updated = resp.json()
    assert updated["status"] == "in_progress"
    assert updated["started_at"] is not None
    assert product_client.get(f"/service-requests/{sr['id']}").json()["status"] == "in_progress"

    resp = product_client.patch(
        f"/work-orders/{work_order['id']}/status", json={"status": "completed"}
    )
    completed = resp.json()
    assert completed["completed_at"] is not None
    assert product_client.get(f"/service-requests/{sr['id']}").json()["status"] == "resolved"


def test_create_work_order_404_for_missing_service_request(product_client):
    resp = product_client.post("/work-orders", json={"service_request_id": 999, "technician_id": 1})
    assert resp.status_code == 404


def test_update_work_order_status_404(product_client):
    resp = product_client.patch("/work-orders/999/status", json={"status": "in_progress"})
    assert resp.status_code == 404


def test_list_work_orders_filters(product_client):
    sr = product_client.post(
        "/service-requests", json={"customer_id": 1, "depot_id": 1, "title": "Noise"}
    ).json()
    product_client.post("/work-orders", json={"service_request_id": sr["id"], "technician_id": 1})

    assert len(product_client.get("/work-orders").json()) == 1
    assert len(product_client.get("/work-orders", params={"technician_id": 1}).json()) == 1
    assert product_client.get("/work-orders", params={"technician_id": 999}).json() == []
    assert len(product_client.get("/work-orders", params={"status": "assigned"}).json()) == 1


# --- Hardening: bad input on POST /service-requests --------------------------


def test_create_service_request_invalid_customer_id_404(product_client):
    resp = product_client.post(
        "/service-requests", json={"customer_id": 999, "depot_id": 1, "title": "Noise"}
    )
    assert resp.status_code == 404
    assert "customer" in resp.json()["detail"]


def test_create_service_request_invalid_depot_id_404(product_client):
    resp = product_client.post(
        "/service-requests", json={"customer_id": 1, "depot_id": 999, "title": "Noise"}
    )
    assert resp.status_code == 404
    assert "depot" in resp.json()["detail"]


def test_create_service_request_invalid_equipment_id_404(product_client):
    resp = product_client.post(
        "/service-requests",
        json={"customer_id": 1, "depot_id": 1, "equipment_id": 999, "title": "Noise"},
    )
    assert resp.status_code == 404
    assert "equipment" in resp.json()["detail"]


def test_create_service_request_equipment_depot_mismatch_422(product_client):
    # Equipment 1 (from product_session) belongs to depot_id=1; naming depot_id=2
    # alongside it is inconsistent client data. Reasoned default: reject (422)
    # rather than silently trusting either field -- see the comment in
    # service_requests.py:create_service_request.
    resp = product_client.post(
        "/service-requests",
        json={"customer_id": 1, "depot_id": 2, "equipment_id": 1, "title": "Noise"},
    )
    assert resp.status_code == 422
    assert "depot" in resp.json()["detail"]


# --- Hardening: bad input on POST /work-orders and status transitions --------


def test_create_work_order_missing_technician_404(product_client):
    sr = product_client.post(
        "/service-requests", json={"customer_id": 1, "depot_id": 1, "title": "Noise"}
    ).json()
    resp = product_client.post(
        "/work-orders", json={"service_request_id": sr["id"], "technician_id": 999}
    )
    assert resp.status_code == 404
    assert "technician" in resp.json()["detail"]


def test_create_work_order_double_dispatch_409(product_client):
    sr = product_client.post(
        "/service-requests", json={"customer_id": 1, "depot_id": 1, "title": "Noise"}
    ).json()
    first = product_client.post(
        "/work-orders", json={"service_request_id": sr["id"], "technician_id": 1}
    )
    assert first.status_code == 201
    first_id = first.json()["id"]

    second = product_client.post(
        "/work-orders", json={"service_request_id": sr["id"], "technician_id": 1}
    )
    assert second.status_code == 409
    assert str(first_id) in second.json()["detail"]


def test_work_order_status_allows_any_transition(product_client):
    """Documents current, intentionally permissive behavior: no transition-graph
    validation, so ops can correct a mistake without a support ticket. A future
    change here should be a conscious decision, not an accidental regression."""
    sr = product_client.post(
        "/service-requests", json={"customer_id": 1, "depot_id": 1, "title": "Noise"}
    ).json()
    wo = product_client.post(
        "/work-orders", json={"service_request_id": sr["id"], "technician_id": 1}
    ).json()

    resp = product_client.patch(f"/work-orders/{wo['id']}/status", json={"status": "completed"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"

    # "backwards" transition, completed -> assigned
    resp = product_client.patch(f"/work-orders/{wo['id']}/status", json={"status": "assigned"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "assigned"

    # setting the same status twice
    resp = product_client.patch(f"/work-orders/{wo['id']}/status", json={"status": "assigned"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "assigned"


# --- Hardening: limit bounds ---------------------------------------------


def test_equipment_limit_zero_rejected(product_client):
    resp = product_client.get("/equipment", params={"limit": 0})
    assert resp.status_code == 422


def test_service_requests_limit_zero_rejected(product_client):
    resp = product_client.get("/service-requests", params={"limit": 0})
    assert resp.status_code == 422


def test_work_orders_limit_negative_rejected(product_client):
    resp = product_client.get("/work-orders", params={"limit": -1})
    assert resp.status_code == 422
