"""Regressionstests für LAUNCH_AUDIT.md P2-13: /auth/register verriet per
Fehlermeldung ("Email already registered"), ob ein Konto mit dieser Email
schon existiert - im Kontrast zu den enumeration-sicheren Flows
(/forgot-password, /resend-verification-public). Email-Kollision liefert
jetzt dieselbe generische Erfolgsantwort wie eine echte Neuregistrierung,
ohne einen zweiten Account anzulegen. Benutzername bleibt bewusst eine
sofortige, konkrete Fehlermeldung (siehe Kommentar in auth.py).

Stil wie test_audit_log_p2_21.py: direkte Funktionsaufrufe, kein TestClient.
"""
from datetime import date

import pytest
from fastapi import BackgroundTasks, HTTPException
from starlette.requests import Request

from api.models.user import User
from api.routes.auth import _REGISTER_GENERIC_RESPONSE, register
from api.schemas.user import UserCreate


def _fake_request() -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/auth/register",
        "headers": [],
        "client": ("testclient", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
        "query_string": b"",
    }
    return Request(scope)


def _payload(**overrides) -> UserCreate:
    defaults = dict(
        email="new@example.com",
        password="hunter2hunter2",
        username="newuser",
        first_name="Test",
        last_name="User",
        birth_date=date(1990, 1, 1),
        terms_accepted=True,
        privacy_accepted=True,
    )
    defaults.update(overrides)
    return UserCreate(**defaults)


def test_register_with_new_email_creates_user_and_returns_generic_message(db):
    result = register(
        request=_fake_request(),
        user=_payload(),
        background_tasks=BackgroundTasks(),
        db=db,
    )

    assert result == _REGISTER_GENERIC_RESPONSE
    assert db.query(User).filter(User.email == "new@example.com").count() == 1


def test_register_with_existing_email_returns_identical_generic_message_without_duplicate(db):
    existing = User(email="taken@example.com", hashed_password="x", username="original")
    db.add(existing)
    db.commit()

    result = register(
        request=_fake_request(),
        user=_payload(email="taken@example.com", username="differentname"),
        background_tasks=BackgroundTasks(),
        db=db,
    )

    # Exakt dieselbe Antwort wie im Erfolgsfall - kein Hinweis, dass die
    # Email schon vergeben war.
    assert result == _REGISTER_GENERIC_RESPONSE
    # Kein zweiter Account mit dieser Email wurde angelegt.
    assert db.query(User).filter(User.email == "taken@example.com").count() == 1
    # Und der neue Username wurde NICHT verbraucht (kein Account angelegt).
    assert db.query(User).filter(User.username == "differentname").count() == 0


def test_register_with_taken_username_still_raises_explicit_error(db):
    db.add(User(email="someone@example.com", hashed_password="x", username="claimed"))
    db.commit()

    with pytest.raises(HTTPException) as exc:
        register(
            request=_fake_request(),
            user=_payload(email="fresh@example.com", username="claimed"),
            background_tasks=BackgroundTasks(),
            db=db,
        )

    assert exc.value.status_code == 400
    assert "Benutzername" in exc.value.detail
