"""Regressionstests für LAUNCH.md P2-17 (Tracking-Lücken im Funnel):
neue Events favorite_added, custom_definition_saved, billing_portal_opened,
sowie die Compare/Custom-Builder-Unterscheidung im Admin-Modus-Chart.

Stil wie test_admin_stats.py/test_account_deletion.py: direkte
Funktionsaufrufe gegen die In-Memory-SQLite-db-Fixture, kein TestClient.
"""
from unittest.mock import MagicMock, patch

import pytest

from api.models.base import Base
from api.models.custom_analysis_definition import CustomAnalysisDefinition
from api.models.favorite import Favorite
from api.models.product_event import ProductEvent
from api.models.symbol import Symbol
from api.models.user import User
from api.routes.admin_stats import _compute_analyses_breakdown
from api.routes.billing import billing_create_portal_session
from api.routes.custom_analysis import create_definition
from api.routes.favorites import create_favorite
from api.schemas.custom_analysis_definition import CustomAnalysisDefinitionCreate


@pytest.fixture
def db_full(db):
    """db-Fixture aus conftest.py (User + ProductEvent) plus die
    Kind-Tabellen, die diese Tests brauchen. PYPL vorab in Symbol gesät -
    seit LAUNCH_AUDIT.md P2-8 validiert create_favorite gegen diese Tabelle."""
    Base.metadata.create_all(
        db.get_bind(),
        tables=[Favorite.__table__, CustomAnalysisDefinition.__table__, Symbol.__table__],
    )
    db.add(Symbol(symbol="PYPL", name="PayPal Holdings", exchange="NASDAQ"))
    db.commit()
    return db


def _make_user(db, **kwargs) -> User:
    defaults = {"email": f"user{id(kwargs)}@example.com", "hashed_password": "x"}
    defaults.update(kwargs)
    user = User(**defaults)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _events(db, event_type: str) -> list[ProductEvent]:
    return db.query(ProductEvent).filter(ProductEvent.event_type == event_type).all()


# ---------------------------------------------------------------------------
# favorite_added
# ---------------------------------------------------------------------------

def test_create_favorite_logs_event(db_full):
    user = _make_user(db_full)
    create_favorite(symbol="pypl", current_user=user, db=db_full)

    events = _events(db_full, "favorite_added")
    assert len(events) == 1
    assert events[0].user_id == user.id
    assert events[0].event_metadata == {"symbol": "PYPL"}


def test_create_favorite_duplicate_still_logs_event(db_full):
    """add_favorite ist idempotent (gibt den bestehenden Favoriten zurück)
    - das Event wird trotzdem pro Request geloggt, nicht nur beim echten
    Insert. Dokumentiert bewusstes Verhalten, keine Überraschung."""
    user = _make_user(db_full)
    create_favorite(symbol="PYPL", current_user=user, db=db_full)
    create_favorite(symbol="PYPL", current_user=user, db=db_full)

    assert len(_events(db_full, "favorite_added")) == 2


# ---------------------------------------------------------------------------
# custom_definition_saved
# ---------------------------------------------------------------------------

def test_create_definition_logs_event(db_full):
    user = _make_user(db_full, plan="pro")
    payload = CustomAnalysisDefinitionCreate(name="Meine Analyse", metrics=[])

    create_definition(payload=payload, current_user=user, db=db_full)

    events = _events(db_full, "custom_definition_saved")
    assert len(events) == 1
    assert events[0].user_id == user.id
    assert events[0].event_metadata == {"metric_count": 0}


# ---------------------------------------------------------------------------
# billing_portal_opened
# ---------------------------------------------------------------------------

def test_billing_portal_session_logs_event(db_full):
    user = _make_user(db_full, stripe_customer_id="cus_test123")

    fake_session = MagicMock(url="https://billing.stripe.com/session/test")
    with patch("stripe.billing_portal.Session.create", return_value=fake_session) as mock_create:
        result = billing_create_portal_session(current_user=user, db=db_full)

    mock_create.assert_called_once()
    assert result == {"url": fake_session.url}

    events = _events(db_full, "billing_portal_opened")
    assert len(events) == 1
    assert events[0].user_id == user.id


def test_billing_portal_session_without_customer_id_raises_and_does_not_log(db_full):
    from fastapi import HTTPException

    user = _make_user(db_full, stripe_customer_id=None)

    with pytest.raises(HTTPException):
        billing_create_portal_session(current_user=user, db=db_full)

    assert _events(db_full, "billing_portal_opened") == []


# ---------------------------------------------------------------------------
# Compare vs. Custom-Builder im Admin-Modus-Chart (analysis_started/source)
# ---------------------------------------------------------------------------

def test_compare_source_bucketed_separately_from_custom_builder(db_full):
    user = _make_user(db_full)
    db_full.add_all(
        [
            ProductEvent(
                event_type="analysis_started",
                user_id=user.id,
                event_metadata={"mode": "custom", "source": "compare", "symbol": "AAPL"},
            ),
            ProductEvent(
                event_type="analysis_started",
                user_id=user.id,
                event_metadata={"mode": "custom", "source": "custom_builder", "symbol": "MSFT"},
            ),
            # Legacy-Event ohne "source"-Feld (vor diesem Fix geloggt) - muss
            # weiterhin als "custom" gezählt werden, nicht verschwinden.
            ProductEvent(
                event_type="analysis_started",
                user_id=user.id,
                event_metadata={"mode": "custom", "symbol": "GOOGL"},
            ),
            ProductEvent(
                event_type="analysis_started",
                user_id=user.id,
                event_metadata={"mode": "full", "symbol": "PYPL"},
            ),
        ]
    )
    db_full.commit()

    breakdown = _compute_analyses_breakdown(db_full)
    counts = {row["mode"]: row["count"] for row in breakdown["by_mode"]}

    assert counts == {"compare": 1, "custom": 2, "full": 1}
