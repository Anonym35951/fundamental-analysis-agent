"""Tests für LAUNCH.md P2-27 (Admin-Löschung von Kunden-Konten) und die
dabei extrahierte gemeinsame Löschlogik user_service.delete_user_account
(vorher inline in auth.py::delete_account).

Stil wie test_admin_stats.py: direkte Funktionsaufrufe mit der In-Memory-
SQLite-db-Fixture, kein TestClient. Die Depends(require_admin)-Verdrahtung
der Route wird daher nicht über HTTP ausgeübt — der Gate selbst ist unten
direkt getestet (test_require_admin_rejects_non_admin), die Verdrahtung ist
per Inspektion identisch mit allen anderen admin_customers-Routen.

Stripe wird hier erstmals im Testlauf gemockt: gepatcht wird bewusst
api.services.user_service.cancel_subscription_immediately (der Name, unter
dem der Service die Funktion aufruft), nicht api.services.stripe_service.
"""
from datetime import datetime

import pytest
import stripe
from fastapi import HTTPException

import api.services.user_service as user_service
from api.core.dependencies import require_admin
from api.models.analysis_history import AnalysisHistory
from api.models.base import Base
from api.models.custom_analysis_definition import CustomAnalysisDefinition
from api.models.customer_note import CustomerNote
from api.models.favorite import Favorite
from api.models.product_event import ProductEvent
from api.models.user import User
from api.routes.admin_customers import delete_customer
from api.services.user_service import StripeCancellationError, delete_user_account


@pytest.fixture
def db_full(db):
    """db-Fixture aus conftest.py (User + ProductEvent) plus die vier
    Kind-Tabellen, die der Löschpfad anfasst. Erst seit der
    with_variant-Anpassung an AnalysisHistory.result_snapshot (JSONB → JSON
    auf SQLite) hier per create_all anlegbar."""
    Base.metadata.create_all(
        db.get_bind(),
        tables=[
            AnalysisHistory.__table__,
            Favorite.__table__,
            CustomAnalysisDefinition.__table__,
            CustomerNote.__table__,
        ],
    )
    return db


def _make_user(db, **kwargs) -> User:
    defaults = {"email": f"user{id(kwargs)}@example.com", "hashed_password": "x"}
    defaults.update(kwargs)
    user = User(**defaults)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _add_child_rows(db, user_id: int):
    """Je eine Zeile in allen vier Kind-Tabellen für diesen User."""
    db.add(
        AnalysisHistory(
            user_id=user_id,
            job_id=f"job-{user_id}",
            symbol="PYPL",
            mode="full",
            status="done",
            result_snapshot={"ok": True},
        )
    )
    db.add(Favorite(user_id=user_id, symbol="PYPL"))
    db.add(CustomAnalysisDefinition(user_id=user_id, name="Meine Analyse", metrics=[]))
    db.add(CustomerNote(user_id=user_id, note="Testnotiz"))
    db.commit()


def _count_rows_for_user(db, user_id: int) -> dict:
    return {
        "user": db.query(User).filter(User.id == user_id).count(),
        "history": db.query(AnalysisHistory).filter(AnalysisHistory.user_id == user_id).count(),
        "favorites": db.query(Favorite).filter(Favorite.user_id == user_id).count(),
        "definitions": db.query(CustomAnalysisDefinition)
        .filter(CustomAnalysisDefinition.user_id == user_id)
        .count(),
        "notes": db.query(CustomerNote).filter(CustomerNote.user_id == user_id).count(),
    }


def _account_deleted_events(db):
    return (
        db.query(ProductEvent)
        .filter(ProductEvent.event_type == "account_deleted")
        .all()
    )


# ---------------------------------------------------------------------------
# Service: delete_user_account
# ---------------------------------------------------------------------------

def test_delete_user_account_removes_user_and_child_rows(db_full, monkeypatch):
    canceled_ids = []
    monkeypatch.setattr(
        user_service,
        "cancel_subscription_immediately",
        lambda sub_id: canceled_ids.append(sub_id),
    )

    target = _make_user(db_full, email="target@x.com", stripe_subscription_id="sub_123")
    other = _make_user(db_full, email="other@x.com")
    # IDs vor der Löschung sichern: nach dem Commit ist die ORM-Instanz
    # detached und Attributzugriffe schlagen fehl (DetachedInstanceError).
    target_id, other_id = target.id, other.id
    _add_child_rows(db_full, target_id)
    _add_child_rows(db_full, other_id)

    delete_user_account(db_full, target, event_metadata={"deleted_by": "self"})

    assert _count_rows_for_user(db_full, target_id) == {
        "user": 0,
        "history": 0,
        "favorites": 0,
        "definitions": 0,
        "notes": 0,
    }
    # Kontrollgruppe: der zweite User und seine Daten bleiben unberührt.
    assert _count_rows_for_user(db_full, other_id) == {
        "user": 1,
        "history": 1,
        "favorites": 1,
        "definitions": 1,
        "notes": 1,
    }
    assert canceled_ids == ["sub_123"]

    events = _account_deleted_events(db_full)
    assert len(events) == 1
    assert events[0].event_metadata == {"deleted_by": "self"}


def test_delete_user_account_stripe_hard_failure_aborts_deletion(db_full, monkeypatch):
    def _boom(sub_id):
        raise RuntimeError("stripe down")

    monkeypatch.setattr(user_service, "cancel_subscription_immediately", _boom)

    target = _make_user(db_full, email="target@x.com", stripe_subscription_id="sub_123")
    _add_child_rows(db_full, target.id)

    with pytest.raises(StripeCancellationError):
        delete_user_account(db_full, target)

    # Nichts gelöscht, kein Event geloggt.
    assert _count_rows_for_user(db_full, target.id)["user"] == 1
    assert _count_rows_for_user(db_full, target.id)["notes"] == 1
    assert _account_deleted_events(db_full) == []


def test_delete_user_account_stripe_invalid_request_continues(db_full, monkeypatch):
    """Abo existiert bei Stripe nicht (mehr) — Löschung läuft trotzdem durch."""

    def _gone(sub_id):
        raise stripe.error.InvalidRequestError("No such subscription", "id")

    monkeypatch.setattr(user_service, "cancel_subscription_immediately", _gone)

    target = _make_user(db_full, email="target@x.com", stripe_subscription_id="sub_gone")
    target_id = target.id

    delete_user_account(db_full, target)

    assert _count_rows_for_user(db_full, target_id)["user"] == 0
    assert len(_account_deleted_events(db_full)) == 1


def test_delete_user_account_without_subscription_never_calls_stripe(db_full, monkeypatch):
    def _fail(sub_id):
        raise AssertionError("cancel_subscription_immediately darf nicht aufgerufen werden")

    monkeypatch.setattr(user_service, "cancel_subscription_immediately", _fail)

    target = _make_user(db_full, email="target@x.com", stripe_subscription_id=None)
    target_id = target.id

    delete_user_account(db_full, target)

    assert _count_rows_for_user(db_full, target_id)["user"] == 0


# ---------------------------------------------------------------------------
# Route: DELETE /admin/customers/{user_id}
# ---------------------------------------------------------------------------

def test_delete_customer_happy_path_logs_admin_metadata(db_full, monkeypatch):
    monkeypatch.setattr(user_service, "cancel_subscription_immediately", lambda sub_id: None)

    admin = _make_user(db_full, email="admin@x.com", plan="admin")
    target = _make_user(db_full, email="kunde@x.com", plan="free")
    target_id = target.id
    _add_child_rows(db_full, target_id)

    delete_customer(user_id=target_id, db=db_full, current_admin=admin)

    assert _count_rows_for_user(db_full, target_id)["user"] == 0
    events = _account_deleted_events(db_full)
    assert len(events) == 1
    assert events[0].event_metadata == {
        "deleted_by": "admin",
        "admin_id": admin.id,
        "admin_email": "admin@x.com",
    }


def test_delete_customer_unknown_id_returns_404(db_full):
    admin = _make_user(db_full, email="admin@x.com", plan="admin")

    with pytest.raises(HTTPException) as exc_info:
        delete_customer(user_id=999999, db=db_full, current_admin=admin)

    assert exc_info.value.status_code == 404


def test_delete_customer_blocks_self_deletion(db_full):
    admin = _make_user(db_full, email="admin@x.com", plan="admin")

    with pytest.raises(HTTPException) as exc_info:
        delete_customer(user_id=admin.id, db=db_full, current_admin=admin)

    assert exc_info.value.status_code == 400
    assert db_full.query(User).filter(User.id == admin.id).count() == 1


def test_delete_customer_blocks_other_admin_accounts(db_full):
    """Defense-in-Depth analog zu ADMIN_ASSIGNABLE_PLANS: Admin-Konten sind
    nur out-of-band (scripts/set_admin.py) verwaltbar."""
    admin = _make_user(db_full, email="admin@x.com", plan="admin")
    other_admin = _make_user(db_full, email="admin2@x.com", plan="admin")

    with pytest.raises(HTTPException) as exc_info:
        delete_customer(user_id=other_admin.id, db=db_full, current_admin=admin)

    assert exc_info.value.status_code == 400
    assert db_full.query(User).filter(User.id == other_admin.id).count() == 1


def test_delete_customer_stripe_failure_returns_502_and_keeps_user(db_full, monkeypatch):
    def _boom(sub_id):
        raise RuntimeError("stripe down")

    monkeypatch.setattr(user_service, "cancel_subscription_immediately", _boom)

    admin = _make_user(db_full, email="admin@x.com", plan="admin")
    target = _make_user(
        db_full, email="kunde@x.com", plan="pro", stripe_subscription_id="sub_123"
    )

    with pytest.raises(HTTPException) as exc_info:
        delete_customer(user_id=target.id, db=db_full, current_admin=admin)

    assert exc_info.value.status_code == 502
    assert db_full.query(User).filter(User.id == target.id).count() == 1
    assert _account_deleted_events(db_full) == []


def test_require_admin_rejects_non_admin(db):
    """Der Auth-Gate der Route: Depends(require_admin) wird im Direct-Call-
    Stil nicht mit ausgeübt, deshalb hier der Gate selbst."""
    non_admin = _make_user(db, email="free@x.com", plan="free")

    with pytest.raises(HTTPException) as exc_info:
        require_admin(current_user=non_admin)

    assert exc_info.value.status_code == 403
