"""EVOLVING.md EV-124: Landingpage-CTA-Attribution über `?src=` am
Registrierungs-Event. Kein neuer öffentlicher Tracking-Endpoint - der
optionale `src`-Wert wird nur als Metadata an das bereits existierende
`user_registered`-Event (api/services/event_service.log_event) gehängt.

Stil wie test_register_enumeration_p2_13.py: direkte Funktionsaufrufe,
kein TestClient.
"""
from datetime import date

import pytest
from fastapi import BackgroundTasks
from starlette.requests import Request

from api.core.rate_limit import limiter
from api.models.base import Base
from api.models.product_event import ProductEvent
from api.models.user import User
from api.routes.auth import register
from api.schemas.user import UserCreate


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    # /auth/register hat ein Limit von 3/minute; alle Tests hier teilen sich
    # denselben Fake-Client ("testclient") und würden sich sonst gegenseitig
    # den Bucket leeren (429 statt des erwarteten Verhaltens) - gleiches
    # Muster wie test_price_history_batch_ev070.py.
    limiter.reset()


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


def _events_db(db):
    Base.metadata.create_all(db.get_bind(), tables=[ProductEvent.__table__])
    return db


def test_register_with_valid_src_attaches_it_to_event_metadata(db):
    _events_db(db)

    register(
        request=_fake_request(),
        user=_payload(src="hero"),
        background_tasks=BackgroundTasks(),
        db=db,
    )

    user = db.query(User).filter(User.email == "new@example.com").one()
    event = db.query(ProductEvent).filter(ProductEvent.event_type == "user_registered").one()
    assert event.user_id == user.id
    assert event.event_metadata == {"src": "hero"}


def test_register_without_src_logs_event_without_metadata(db):
    _events_db(db)

    register(
        request=_fake_request(),
        user=_payload(),
        background_tasks=BackgroundTasks(),
        db=db,
    )

    event = db.query(ProductEvent).filter(ProductEvent.event_type == "user_registered").one()
    assert event.event_metadata is None


def test_register_with_unknown_src_is_silently_dropped_not_rejected():
    """Ein manipulierter/unbekannter src-Wert darf die Registrierung nicht
    blockieren (422) - er ist rein informativ, nie sicherheitsrelevant."""
    payload = _payload(src="not-a-real-source")

    assert payload.src is None
