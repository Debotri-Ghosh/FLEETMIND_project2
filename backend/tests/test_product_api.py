"""Coverage for the product-layer routers: equipment, customers, technicians,
service-requests, work-orders. See tests/conftest.py:product_client for the
seed data (1 customer, 1 technician, 3 equipment rows — one linked to an
experiment with a HIGH agent_decision, one unlinked, one retired)."""


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
    resp = product_client.post(
        "/work-orders", json={"service_request_id": 999, "technician_id": 1}
    )
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
