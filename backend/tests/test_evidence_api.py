"""Tests for GET /service-requests/{id}/evidence.

The evidence package is what makes an AI-filed request checkable instead of
trusted, so these assert it actually carries the underlying signal, a real-Hz
spectrum, and the corroborating defect frequency -- and that it refuses
honestly when there is nothing to show.
"""

from fleetmind.agents.sync_equipment_status import sync_equipment_status


def _ai_request_id(product_client, product_session) -> int:
    sync_equipment_status(product_session)
    product_session.commit()
    rows = product_client.get("/service-requests", params={"ai_flagged": True}).json()
    assert rows, "fixture should have produced one AI-filed request"
    return rows[0]["id"]


def test_evidence_returns_the_triggering_window_and_signal(product_client, product_session):
    rid = _ai_request_id(product_client, product_session)

    resp = product_client.get(f"/service-requests/{rid}/evidence")

    assert resp.status_code == 200
    body = resp.json()
    assert body["service_request_id"] == rid
    assert body["window_id"] > 0
    # the fixture window carries 8 raw samples
    assert len(body["raw_values"]) == 8
    assert body["sampling_rate_hz"] == 12000


def test_evidence_includes_a_real_hz_spectrum(product_client, product_session):
    rid = _ai_request_id(product_client, product_session)

    body = product_client.get(f"/service-requests/{rid}/evidence").json()

    assert len(body["spectrum_freqs_hz"]) == len(body["spectrum_magnitudes"])
    assert len(body["spectrum_freqs_hz"]) > 0
    # frequencies must be Hz, ascending, starting at DC
    assert body["spectrum_freqs_hz"][0] == 0.0
    assert body["spectrum_freqs_hz"] == sorted(body["spectrum_freqs_hz"])


def test_evidence_annotates_the_matching_defect_frequency(product_client, product_session):
    """The fixture's prediction is InnerRace, so the band must be BPFI."""
    rid = _ai_request_id(product_client, product_session)

    body = product_client.get(f"/service-requests/{rid}/evidence").json()

    matched = body["matched_frequency"]
    assert matched is not None
    assert matched["name"] == "BPFI"
    assert matched["band_low_hz"] < matched["hz"] < matched["band_high_hz"]
    # and all four are offered for context
    assert set(body["all_characteristic_frequencies"]) == {"BPFI", "BPFO", "BSF", "FTF"}


def test_evidence_carries_model_confidence_and_version(product_client, product_session):
    rid = _ai_request_id(product_client, product_session)

    body = product_client.get(f"/service-requests/{rid}/evidence").json()

    assert body["predicted_label"] == "InnerRace"
    assert body["model_confidence"] == 0.9  # set by the product_session fixture
    assert body["model_version"] == "v1"


def test_evidence_carries_the_diagnostic_agents_own_stats(product_client, product_session):
    """These are recomputed with the same diagnostic_agent function that made
    the decision -- so they must be present and self-consistent."""
    rid = _ai_request_id(product_client, product_session)

    body = product_client.get(f"/service-requests/{rid}/evidence").json()

    assert body["signal_rms"] > 0
    assert body["signal_peak"] > 0
    assert 0.0 <= body["diagnosis_confidence"] <= 1.0
    assert body["diagnosis"].startswith("InnerRace")
    assert body["risk_score"] is not None
    assert "[HIGH]" in body["decision_text"]


def test_evidence_404s_for_a_request_that_does_not_exist(product_client):
    resp = product_client.get("/service-requests/99999/evidence")

    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


def test_evidence_explains_why_a_customer_filed_request_has_none(product_client):
    """A customer-filed request legitimately has no model evidence. That must
    read as an explanation, not a bare 'not found'."""
    created = product_client.post(
        "/service-requests",
        json={"customer_id": 1, "depot_id": 1, "title": "Heard a noise"},
    ).json()

    resp = product_client.get(f"/service-requests/{created['id']}/evidence")

    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert "no agent decision" in detail
    assert "customer-filed" in detail
