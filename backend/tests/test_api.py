def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"


def test_list_depots(client):
    resp = client.get("/depots")
    assert resp.status_code == 200
    depots = resp.json()
    assert len(depots) == 4
    assert depots[0]["name"] == "Plant A (CWRU 0 HP)"
    assert depots[0]["n_experiments"] == 1


def test_fault_distribution(client):
    resp = client.get("/stats/fault-distribution")
    assert resp.status_code == 200
    rows = resp.json()
    assert rows == [{"label": "InnerRace", "n_windows": 1}]


def test_list_experiments_filter(client):
    resp = client.get("/experiments", params={"fault_type": "Normal"})
    assert resp.status_code == 200
    assert resp.json() == []

    resp = client.get("/experiments", params={"fault_type": "InnerRace"})
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_get_experiment_404(client):
    assert client.get("/experiments/999").status_code == 404


def test_windows_for_experiment(client):
    resp = client.get("/experiments/1/windows")
    assert resp.status_code == 200
    windows = resp.json()
    assert len(windows) == 1
    assert windows[0]["label"] == "InnerRace"
    assert windows[0]["window_index"] == 0


def test_signal_window_detail(client):
    resp = client.get("/signal-window/1")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["raw_values"]) == 8
    assert body["raw_values"][0] == 0.1


def test_signal_window_404(client):
    assert client.get("/signal-window/999").status_code == 404
