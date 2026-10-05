"""Tests for JWT auth and role-based access control.

The brief for this layer was "protect routes by role at the API layer, not
just hidden UI", so these tests never look at the frontend. They call the
endpoints directly as each role and assert on what comes back.

Two of them are worth reading first:

  `test_every_route_refuses_an_anonymous_caller` enumerates the application's
  own route table rather than a hand-written list, so a router added later
  without a guard fails this suite instead of shipping open.

  `test_a_customer_cannot_widen_its_scope_with_a_query_parameter` is the
  attack the scoping design exists to stop.
"""

from __future__ import annotations

from datetime import timedelta

import jwt
import pytest
from conftest import TEST_PASSWORD, authorize, make_user
from fastapi.routing import APIRoute

from fleetmind.api.main import create_app
from fleetmind.auth.passwords import (
    MAX_PASSWORD_BYTES,
    PasswordTooLong,
    hash_password,
    verify_password,
)
from fleetmind.auth.tokens import ALGORITHM, create_access_token, get_secret
from fleetmind.db.models import User

# Routes that are open by design. Everything else must refuse an anonymous
# caller; this list is deliberately tiny and any addition to it should be
# argued for in review.
PUBLIC_PATHS = {
    "/livez",  # liveness probe for container healthchecks, discloses nothing
    "/auth/login",  # the route that issues tokens cannot itself require one
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
}


def _concrete(path: str) -> str:
    """Turn '/equipment/{equipment_id}' into a requestable '/equipment/1'."""
    parts = []
    for segment in path.split("/"):
        parts.append("1" if segment.startswith("{") and segment.endswith("}") else segment)
    return "/".join(parts)


def _all_routes(app) -> list[tuple[str, str]]:
    """Every (method, path) the application actually serves.

    `app.routes` is NOT a flat list of APIRoute. This FastAPI version wraps
    each `include_router()` call in an internal `_IncludedRouter` object that
    exposes its children only via `.original_router.routes` -- so naively
    filtering `app.routes` for APIRoute finds just the handlers declared
    directly on the app. In this project that was 2 of 34 routes, and after
    subtracting the public ones the route-walking auth test below was probing
    exactly ONE endpoint while appearing to cover everything.

    Both shapes are handled so this keeps working whichever way a future
    FastAPI version organises the table.
    """
    found: list[tuple[str, str]] = []

    def walk(routes) -> None:
        for route in routes:
            if isinstance(route, APIRoute):
                for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
                    found.append((method, route.path))
            else:
                nested = getattr(getattr(route, "original_router", None), "routes", None)
                if nested is None:
                    nested = getattr(route, "routes", None)
                if nested:
                    walk(nested)

    walk(app.routes)
    return found


def test_every_route_refuses_an_anonymous_caller(anon_client):
    """Enumerated from the app's own route table, not a hand-written list.

    A router added later without a guard shows up here as a failure rather
    than silently shipping open, which is the whole reason this is written
    against `app.routes` instead of against a list of URLs.
    """
    app = create_app()
    unprotected: list[str] = []

    for method, path in _all_routes(app):
        if path in PUBLIC_PATHS:
            continue
        response = anon_client.request(method, _concrete(path), json={})
        # 401 is the correct answer. A 422 would mean the body was validated
        # before auth ran, which leaks the schema; anything 2xx means the
        # route is open.
        if response.status_code != 401:
            unprotected.append(f"{method} {path} -> {response.status_code}")

    assert not unprotected, "routes reachable without a token: " + "; ".join(unprotected)


def test_the_route_walk_actually_finds_the_whole_api():
    """Guards the guard: a reachability test that reaches nothing passes.

    The test above spent this project's whole history probing a single
    endpoint because `app.routes` is not flat. Pinning a realistic floor means
    that if a future FastAPI release changes the table's shape again, this
    fails loudly instead of quietly going vacuous.
    """
    routes = _all_routes(create_app())
    paths = {p for _, p in routes}

    assert len(routes) >= 30, f"route walk found only {len(routes)} routes -- it has broken"
    # A spread across routers, so finding one router's routes is not enough.
    for expected in (
        "/auth/login",
        "/equipment",
        "/service-requests",
        "/work-orders",
        "/replay/status",
        "/stats/product-summary",
        "/signal-window/{window_id}/spectrum",
    ):
        assert expected in paths, f"route walk missed {expected}"


def test_livez_is_open_but_health_is_not(anon_client):
    """/health reports database state, so it is behind auth like everything else."""
    alive = anon_client.get("/livez")
    assert alive.status_code == 200
    assert alive.json() == {"status": "alive"}
    # and it must not leak anything a probe does not need
    assert "version" not in alive.text and "database" not in alive.text

    assert anon_client.get("/health").status_code == 401


# ── login ──────────────────────────────────────────────────────────────


def test_login_returns_a_token_and_the_callers_scope(
    product_client, product_session, test_password_hash
):
    make_user(
        product_session,
        test_password_hash,
        email="op@fleetmind.example.com",
        role="operator",
        depot_id=2,
        full_name="Site Operator",
    )

    resp = product_client.post(
        "/auth/login", json={"email": "op@fleetmind.example.com", "password": TEST_PASSWORD}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["user"]["role"] == "operator"
    assert body["user"]["depot_id"] == 2
    assert body["user"]["site_name"] == "Plant B (CWRU 1 HP)"
    # the hash must never cross the wire
    assert "password" not in resp.text and "hashed" not in resp.text


def test_login_is_case_insensitive_on_email(product_client, product_session, test_password_hash):
    make_user(
        product_session, test_password_hash, email="mixed@fleetmind.example.com", role="admin"
    )
    resp = product_client.post(
        "/auth/login", json={"email": "  Mixed@Fleetmind.Example.Com  ", "password": TEST_PASSWORD}
    )
    assert resp.status_code == 200


@pytest.mark.parametrize(
    ("email", "password"),
    [
        ("admin@fleetmind.example.com", "wrong-password"),
        ("nobody@fleetmind.example.com", TEST_PASSWORD),
    ],
    ids=["wrong password", "unknown email"],
)
def test_bad_credentials_give_one_indistinguishable_error(product_client, email, password):
    """Wrong password and unknown account must look identical to the caller.

    Any difference in status or wording is an account-enumeration oracle.
    """
    resp = product_client.post("/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "incorrect email or password"


def test_a_deactivated_account_cannot_log_in(product_client, product_session, test_password_hash):
    user = make_user(
        product_session, test_password_hash, email="gone@fleetmind.example.com", role="admin"
    )
    user.is_active = False
    product_session.commit()

    resp = product_client.post(
        "/auth/login", json={"email": "gone@fleetmind.example.com", "password": TEST_PASSWORD}
    )
    assert resp.status_code == 401


def test_deactivating_an_account_takes_effect_on_the_next_request(
    product_client, product_session, test_password_hash
):
    """Not when their token expires -- the user row is re-read every request."""
    user = make_user(
        product_session, test_password_hash, email="live@fleetmind.example.com", role="admin"
    )
    authorize(product_client, user)
    assert product_client.get("/customers").status_code == 200

    user.is_active = False
    product_session.commit()

    assert product_client.get("/customers").status_code == 401


def test_changing_a_role_in_the_database_invalidates_the_old_token(
    product_client, product_session, test_password_hash
):
    user = make_user(
        product_session, test_password_hash, email="promoted@fleetmind.example.com", role="admin"
    )
    authorize(product_client, user)  # token says admin

    user.role = "operator"
    user.depot_id = 1
    product_session.commit()

    resp = product_client.get("/customers")
    assert resp.status_code == 401
    assert "sign in again" in resp.json()["detail"]


# ── tokens ─────────────────────────────────────────────────────────────


def test_an_expired_token_is_refused(product_client, product_session, test_password_hash):
    user = make_user(
        product_session, test_password_hash, email="exp@fleetmind.example.com", role="admin"
    )
    stale = create_access_token(user.id, "admin", expires_in=timedelta(seconds=-1))
    product_client.headers["Authorization"] = f"Bearer {stale}"
    assert product_client.get("/customers").status_code == 401


def test_a_token_signed_with_another_key_is_refused(
    product_client, product_session, test_password_hash
):
    user = make_user(
        product_session, test_password_hash, email="forge@fleetmind.example.com", role="admin"
    )
    forged = jwt.encode({"sub": str(user.id), "role": "admin"}, "not-the-secret", algorithm="HS256")
    product_client.headers["Authorization"] = f"Bearer {forged}"
    assert product_client.get("/customers").status_code == 401


def test_an_alg_none_token_is_refused(product_client, product_session, test_password_hash):
    """The classic JWT bypass: an unsigned token claiming alg 'none'."""
    user = make_user(
        product_session, test_password_hash, email="none@fleetmind.example.com", role="admin"
    )
    unsigned = jwt.encode({"sub": str(user.id), "role": "admin"}, key="", algorithm="none")
    product_client.headers["Authorization"] = f"Bearer {unsigned}"
    assert product_client.get("/customers").status_code == 401


def test_a_token_for_a_nonexistent_user_is_refused(product_client):
    ghost = create_access_token(999999, "admin")
    product_client.headers["Authorization"] = f"Bearer {ghost}"
    assert product_client.get("/customers").status_code == 401


def test_a_token_missing_its_claims_is_refused(product_client):
    naked = jwt.encode({"hello": "world"}, get_secret(), algorithm=ALGORITHM)
    product_client.headers["Authorization"] = f"Bearer {naked}"
    assert product_client.get("/customers").status_code == 401


def test_a_malformed_authorization_header_is_refused(product_client):
    product_client.headers["Authorization"] = "Bearer not-a-jwt"
    assert product_client.get("/customers").status_code == 401


# ── customer scope ─────────────────────────────────────────────────────


def test_a_customer_sees_only_its_own_equipment(customer_client, other_customer_client):
    mine = customer_client.get("/equipment").json()
    theirs = other_customer_client.get("/equipment").json()

    assert len(mine) == 3  # the fixture's three machines, all customer 1's
    assert {e["customer_id"] for e in mine} == {1}
    assert theirs == []  # customer 2 owns nothing in this fixture


def test_a_customer_cannot_widen_its_scope_with_a_query_parameter(other_customer_client):
    """?customer_id=1 must not reveal customer 1's machines to customer 2.

    This is the attack the scoping design exists to stop: the caller's scope
    is derived from their user row and applied before any client-supplied
    filter, so a query parameter can only ever narrow the result.
    """
    assert other_customer_client.get("/equipment", params={"customer_id": 1}).json() == []
    assert other_customer_client.get("/service-requests", params={"customer_id": 1}).json() == []


def test_another_customers_equipment_is_404_not_403(other_customer_client):
    """404 so a customer cannot enumerate which equipment ids exist."""
    resp = other_customer_client.get("/equipment/1")
    assert resp.status_code == 404


def test_a_customer_cannot_file_a_request_for_someone_else(customer_client):
    resp = customer_client.post(
        "/service-requests",
        json={"customer_id": 2, "depot_id": 1, "title": "not mine to file"},
    )
    assert resp.status_code == 403
    assert "only file service requests for itself" in resp.json()["detail"]


def test_a_customer_may_still_file_for_itself(customer_client):
    resp = customer_client.post(
        "/service-requests",
        json={"customer_id": 1, "depot_id": 1, "title": "Unusual noise"},
    )
    assert resp.status_code == 201


def test_a_customer_sees_only_itself_in_the_customer_roster(other_customer_client):
    rows = other_customer_client.get("/customers").json()
    assert [c["id"] for c in rows] == [2]


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/work-orders"),
        ("POST", "/work-orders"),
        ("GET", "/technicians"),
        ("GET", "/replay/status"),
        ("POST", "/replay/step"),
    ],
)
def test_a_customer_is_forbidden_from_operator_surfaces(customer_client, method, path):
    resp = customer_client.request(method, path, json={})
    assert resp.status_code == 403
    assert "customer" in resp.json()["detail"]


def test_a_customer_cannot_spend_money_on_a_narrative(customer_client, product_session):
    """Generating a write-up costs a real API call, so it is operator/admin only.

    A customer may still *read* one about their own machine; only generation
    is restricted, so a tenant cannot run up a bill.
    """
    from fleetmind.agents.sync_equipment_status import sync_equipment_status

    sync_equipment_status(product_session)
    product_session.commit()

    resp = customer_client.post("/service-requests/1/narrative")
    assert resp.status_code == 403
    assert "customer" in resp.json()["detail"]


# ── operator scope ─────────────────────────────────────────────────────


def test_an_operator_sees_only_its_own_site(operator_client, other_operator_client):
    at_site_1 = operator_client.get("/equipment").json()
    at_site_3 = other_operator_client.get("/equipment").json()

    assert len(at_site_1) == 3
    assert {e["depot_id"] for e in at_site_1} == {1}
    assert at_site_3 == []


def test_an_operator_cannot_widen_its_site_with_a_query_parameter(other_operator_client):
    assert other_operator_client.get("/equipment", params={"depot_id": 1}).json() == []


def test_an_operator_cannot_read_another_sites_equipment(other_operator_client):
    assert other_operator_client.get("/equipment/1").status_code == 404


def test_an_operator_cannot_open_a_request_at_another_site(other_operator_client):
    resp = other_operator_client.post(
        "/service-requests",
        json={"customer_id": 1, "depot_id": 1, "title": "not my site"},
    )
    assert resp.status_code == 404


@pytest.mark.parametrize("path", ["/replay/status", "/replay/step", "/replay/reset"])
def test_replay_is_admin_only(operator_client, path):
    """Replay rewrites predictions, decisions and equipment status for every
    site, so an operator scoped to one site must not be able to trigger it."""
    method = "GET" if path.endswith("status") else "POST"
    resp = operator_client.request(method, path, json={})
    assert resp.status_code == 403
    assert "operator" in resp.json()["detail"]


def test_an_admin_sees_everything(product_client):
    assert len(product_client.get("/equipment").json()) == 3
    assert product_client.get("/technicians").status_code == 200
    assert product_client.get("/replay/status").status_code == 200


# ── /auth/me ───────────────────────────────────────────────────────────


def test_me_reports_the_callers_scope(operator_client):
    body = operator_client.get("/auth/me").json()
    assert body["role"] == "operator"
    assert body["depot_id"] == 1
    assert body["site_name"] == "Plant A (CWRU 0 HP)"
    assert body["customer_id"] is None


def test_me_requires_a_token(anon_client):
    assert anon_client.get("/auth/me").status_code == 401


# ── password handling ──────────────────────────────────────────────────


def test_hashing_is_salted_so_two_identical_passwords_differ():
    assert hash_password("same") != hash_password("same")


def test_verification_round_trips():
    hashed = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", hashed)
    assert not verify_password("correct horse battery stapl", hashed)


def test_an_overlong_password_is_rejected_not_truncated():
    """bcrypt silently ignores bytes past 72; two long passwords sharing a
    prefix would otherwise become interchangeable."""
    with pytest.raises(PasswordTooLong):
        hash_password("a" * (MAX_PASSWORD_BYTES + 1))


def test_verification_of_a_corrupt_hash_is_false_not_an_exception():
    assert verify_password("anything", "not-a-bcrypt-hash") is False


# ── the database invariant ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [
        {"role": "operator"},  # operator with no site
        {"role": "customer"},  # customer with no company
        {"role": "admin", "depot_id": 1},  # admin carrying a scope
        {"role": "operator", "depot_id": 1, "customer_id": 1},  # both scopes
        {"role": "superuser"},  # not a role at all
    ],
    ids=["operator without site", "customer without company", "scoped admin", "both", "bad role"],
)
def test_a_half_configured_login_cannot_be_inserted(product_session, test_password_hash, kwargs):
    """The CHECK constraint is the backstop.

    An operator with no depot_id would pass `scope_by_depot` with
    `column == None`, matching nothing -- or, with a different bug, matching
    everything. Making it unrepresentable in the database is cheaper than
    trusting every future seed script to get it right.
    """
    from sqlalchemy.exc import IntegrityError

    product_session.add(
        User(email="bad@fleetmind.example.com", hashed_password=test_password_hash, **kwargs)
    )
    with pytest.raises(IntegrityError):
        product_session.commit()
    product_session.rollback()


# ── scoped aggregates ──────────────────────────────────────────────────


def test_product_summary_counts_only_the_callers_own_rows(
    product_client, customer_client, other_customer_client, other_operator_client
):
    """A KPI tile that counted globally would quietly undo the row scoping.

    /stats/product-summary is operational data, unlike fault-distribution and
    model-performance, which are research results over the public archive and
    are deliberately the same number for everyone.
    """
    everything = product_client.get("/stats/product-summary").json()
    assert sum(everything["equipment_by_status"].values()) == 3  # the fixture's three

    mine = customer_client.get("/stats/product-summary").json()
    assert sum(mine["equipment_by_status"].values()) == 3

    for blind in (other_customer_client, other_operator_client):
        summary = blind.get("/stats/product-summary").json()
        assert summary["equipment_by_status"] == {}
        assert summary["open_requests_by_priority"] == {}
        assert summary["ai_flagged_open"] == 0
        assert summary["active_work_orders"] == 0


def test_research_stats_are_the_same_for_every_role(product_client, customer_client):
    """Fault distribution is a property of the CWRU archive, not of a tenant."""
    as_admin = product_client.get("/stats/fault-distribution").json()
    as_customer = customer_client.get("/stats/fault-distribution").json()
    assert as_admin == as_customer


def test_work_orders_are_scoped_through_their_service_request(
    product_client, operator_client, other_operator_client
):
    """Work orders carry no depot_id, so the scope is reached via the request.

    Without that join an operator would see every site's dispatches, which is
    exactly the leak this asserts against.
    """
    created = product_client.post(
        "/service-requests",
        json={"customer_id": 1, "depot_id": 1, "equipment_id": 1, "title": "Bearing noise"},
    )
    assert created.status_code == 201
    dispatched = product_client.post(
        "/work-orders",
        json={"service_request_id": created.json()["id"], "technician_id": 1},
    )
    assert dispatched.status_code == 201
    work_order_id = dispatched.json()["id"]

    # the site-1 operator owns it
    assert [w["id"] for w in operator_client.get("/work-orders").json()] == [work_order_id]
    assert operator_client.get(f"/work-orders/{work_order_id}").status_code == 200

    # the site-3 operator must neither list nor fetch nor move it
    assert other_operator_client.get("/work-orders").json() == []
    assert other_operator_client.get(f"/work-orders/{work_order_id}").status_code == 404
    moved = other_operator_client.patch(
        f"/work-orders/{work_order_id}/status", json={"status": "completed"}
    )
    assert moved.status_code == 404
