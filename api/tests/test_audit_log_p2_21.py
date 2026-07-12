"""Regressionstests für LAUNCH.md P2-21 (Security-/Audit-Log für
Auth-Ereignisse: Failed Logins, Admin-CRM-Zugriffe).

Stil wie test_account_deletion.py: direkte Funktionsaufrufe gegen die
In-Memory-SQLite-db-Fixture, kein TestClient. @limiter.limit verlangt ein
echtes starlette.requests.Request-Objekt, siehe _fake_request().
"""
import pytest
from fastapi import HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from starlette.requests import Request

from api.core.security import hash_password
from api.models.product_event import ProductEvent
from api.models.user import User
from api.routes.admin_customers import get_customer
from api.routes.auth import login


def _make_user(db, **kwargs) -> User:
    defaults = {
        "email": f"user{id(kwargs)}@example.com",
        "hashed_password": hash_password("correct-horse-battery-staple"),
        "email_verified": True,
    }
    defaults.update(kwargs)
    user = User(**defaults)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _fake_request(method: str = "GET", path: str = "/") -> Request:
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [],
        "client": ("testclient", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
        "query_string": b"",
    }
    return Request(scope)


def _events(db, event_type: str) -> list[ProductEvent]:
    return db.query(ProductEvent).filter(ProductEvent.event_type == event_type).all()


# ---------------------------------------------------------------------------
# login_failed
# ---------------------------------------------------------------------------

def test_login_with_unknown_user_logs_event_without_pii(db):
    form = OAuth2PasswordRequestForm(username="nobody@example.com", password="whatever")

    with pytest.raises(HTTPException) as exc:
        login(request=_fake_request(), form_data=form, db=db)
    assert exc.value.status_code == 401

    events = _events(db, "login_failed")
    assert len(events) == 1
    assert events[0].user_id is None
    assert events[0].event_metadata == {"reason": "unknown_user"}


def test_login_with_wrong_password_logs_event_with_user_id(db):
    user = _make_user(db, email="target@example.com")
    form = OAuth2PasswordRequestForm(username="target@example.com", password="wrong-password")

    with pytest.raises(HTTPException) as exc:
        login(request=_fake_request(), form_data=form, db=db)
    assert exc.value.status_code == 401

    events = _events(db, "login_failed")
    assert len(events) == 1
    assert events[0].user_id == user.id
    assert events[0].event_metadata == {"reason": "wrong_password"}


def test_login_with_correct_password_does_not_log_failed_event(db):
    _make_user(db, email="ok@example.com")
    form = OAuth2PasswordRequestForm(username="ok@example.com", password="correct-horse-battery-staple")

    result = login(request=_fake_request(), form_data=form, db=db)

    assert "access_token" in result
    assert _events(db, "login_failed") == []


# ---------------------------------------------------------------------------
# admin_customer_viewed
# ---------------------------------------------------------------------------

def test_get_customer_logs_admin_access(db):
    admin = _make_user(db, email="admin@example.com", plan="admin")
    target = _make_user(db, email="customer@example.com")

    result = get_customer(request=_fake_request(), user_id=target.id, db=db, current_admin=admin)

    assert result.id == target.id
    events = _events(db, "admin_customer_viewed")
    assert len(events) == 1
    assert events[0].user_id == admin.id
    assert events[0].event_metadata == {"viewed_user_id": target.id}


def test_get_customer_unknown_id_does_not_log(db):
    admin = _make_user(db, email="admin2@example.com", plan="admin")

    with pytest.raises(HTTPException) as exc:
        get_customer(request=_fake_request(), user_id=999999, db=db, current_admin=admin)
    assert exc.value.status_code == 404

    assert _events(db, "admin_customer_viewed") == []
