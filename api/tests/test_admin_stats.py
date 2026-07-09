"""Regressionstests für LAUNCH.md P1-6 (Churn zeigte systematisch ~0),
P1-7 (MRR hartkodiert/überzeichnet mehrere Fälle) und P2-16 (interne
Accounts verfälschen Aktivitätsmetriken).

Nutzt eine In-Memory-SQLite-DB (siehe conftest.py::db) statt Mocks - die zu
testende Logik besteht fast ausschließlich aus SQL-Aggregationen
(func.count/distinct/filter), die sich nur gegen eine echte (wenn auch
leichtgewichtige) DB sinnvoll prüfen lassen.
"""
from datetime import datetime, timedelta

from api.models.product_event import ProductEvent
from api.models.user import User
from api.routes.admin_stats import (
    _compute_subscription_stats,
    _distinct_user_count,
    _monthly_price_for_interval,
    PRO_MONTHLY_PRICE_EUR,
    PRO_YEARLY_MONTHLY_EQUIVALENT_EUR,
)


def _make_user(db, **kwargs) -> User:
    defaults = {"email": f"user{id(kwargs)}@example.com", "hashed_password": "x"}
    defaults.update(kwargs)
    user = User(**defaults)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _log_event(db, event_type: str, user_id: int, days_ago: float = 0, metadata=None):
    event = ProductEvent(
        event_type=event_type,
        user_id=user_id,
        event_metadata=metadata,
        created_at=datetime.utcnow() - timedelta(days=days_ago),
    )
    db.add(event)
    db.commit()


# ---------------------------------------------------------------------------
# _monthly_price_for_interval - reine Funktion, kein DB-Zugriff nötig
# ---------------------------------------------------------------------------

def test_monthly_price_for_interval_month():
    assert _monthly_price_for_interval("month") == PRO_MONTHLY_PRICE_EUR


def test_monthly_price_for_interval_year():
    assert _monthly_price_for_interval("year") == PRO_YEARLY_MONTHLY_EQUIVALENT_EUR


def test_monthly_price_for_interval_unknown_returns_none():
    """Vorheriger Bug: None fiel in den else-Zweig und wurde wie 'month'
    behandelt (LAUNCH.md P1-7c)."""
    assert _monthly_price_for_interval(None) is None


# ---------------------------------------------------------------------------
# P1-7: MRR-Bucketing nach billing_status/billing_interval
# ---------------------------------------------------------------------------

def test_mrr_counts_only_active_monthly_and_yearly(db):
    _make_user(db, email="a@x.com", hashed_password="x", plan="pro", billing_status="active", billing_interval="month")
    _make_user(db, email="b@x.com", hashed_password="x", plan="pro", billing_status="active", billing_interval="year")

    stats = _compute_subscription_stats(db)

    assert stats["monthly_subscriptions"] == 1
    assert stats["yearly_subscriptions"] == 1
    assert stats["active_pro_subscriptions"] == 2
    assert stats["mrr_eur"] == round(PRO_MONTHLY_PRICE_EUR + PRO_YEARLY_MONTHLY_EQUIVALENT_EUR, 2)


def test_mrr_excludes_past_due_and_canceling_but_reports_at_risk(db):
    """Vorheriger Bug: past_due/canceling zählten als volle MRR (LAUNCH.md
    P1-7b)."""
    _make_user(db, email="active@x.com", hashed_password="x", plan="pro", billing_status="active", billing_interval="month")
    _make_user(db, email="pastdue@x.com", hashed_password="x", plan="pro", billing_status="past_due", billing_interval="month")
    _make_user(db, email="canceling@x.com", hashed_password="x", plan="pro", billing_status="canceling", billing_interval="year")

    stats = _compute_subscription_stats(db)

    # Nur der aktive Monats-Abonnent zählt in mrr_eur / active_pro_subscriptions.
    assert stats["mrr_eur"] == PRO_MONTHLY_PRICE_EUR
    assert stats["active_pro_subscriptions"] == 1
    # past_due + canceling laufen getrennt in at_risk, nicht in mrr_eur.
    assert stats["at_risk_subscriptions"] == 2
    assert stats["at_risk_mrr_eur"] == round(PRO_MONTHLY_PRICE_EUR + PRO_YEARLY_MONTHLY_EQUIVALENT_EUR, 2)


def test_mrr_excludes_canceled_entirely(db):
    _make_user(db, email="canceled@x.com", hashed_password="x", plan="pro", billing_status="canceled", billing_interval="month")

    stats = _compute_subscription_stats(db)

    assert stats["mrr_eur"] == 0
    assert stats["active_pro_subscriptions"] == 0
    assert stats["at_risk_subscriptions"] == 0


def test_unknown_billing_interval_is_not_silently_counted_as_monthly(db):
    """Vorheriger Bug: billing_interval=None fiel in den else-Zweig der
    Preis-Zuordnung und wurde mit 50 € als monatlich gezählt (LAUNCH.md
    P1-7c)."""
    _make_user(db, email="unknown@x.com", hashed_password="x", plan="pro", billing_status="active", billing_interval=None)

    stats = _compute_subscription_stats(db)

    assert stats["mrr_eur"] == 0
    assert stats["monthly_subscriptions"] == 0
    assert stats["active_pro_subscriptions"] == 0
    assert stats["unknown_billing_interval_count"] == 1


# ---------------------------------------------------------------------------
# P1-6: Churn zählt subscription_deleted, nicht subscription_status_changed
# ---------------------------------------------------------------------------

def test_churn_counts_subscription_deleted(db):
    user = _make_user(db, email="churned@x.com", hashed_password="x", plan="free", billing_status="canceled")
    _log_event(db, "subscription_deleted", user.id, days_ago=5)

    stats = _compute_subscription_stats(db)

    assert stats["churned_last_30d"] == 1


def test_churn_does_not_count_status_changed_to_canceling(db):
    """Vorheriger Bug: die Query zählte subscription_status_changed mit
    to='canceled'. Der normale Kündigungsweg (zum Periodenende) erreicht
    aber nur to='canceling' und danach subscription_deleted - die alte
    Query zeigte für diesen (häufigsten) Fall immer 0 (LAUNCH.md P1-6)."""
    user = _make_user(db, email="canceling@x.com", hashed_password="x", plan="pro", billing_status="canceling")
    _log_event(
        db,
        "subscription_status_changed",
        user.id,
        days_ago=5,
        metadata={"from": "active", "to": "canceling"},
    )

    stats = _compute_subscription_stats(db)

    assert stats["churned_last_30d"] == 0


def test_churn_only_counts_events_within_30_days(db):
    user = _make_user(db, email="old@x.com", hashed_password="x", plan="free", billing_status="canceled")
    _log_event(db, "subscription_deleted", user.id, days_ago=45)

    stats = _compute_subscription_stats(db)

    assert stats["churned_last_30d"] == 0


def test_cancellation_requested_tracked_separately_from_churn(db):
    user = _make_user(db, email="requested@x.com", hashed_password="x", plan="pro", billing_status="canceling")
    _log_event(db, "subscription_cancel_requested", user.id, days_ago=2, metadata={"reason": "zu teuer"})

    stats = _compute_subscription_stats(db)

    assert stats["cancellations_requested_last_30d"] == 1
    assert stats["churned_last_30d"] == 0


# ---------------------------------------------------------------------------
# P2-16: interne Accounts (admin/friends) fließen nicht in Aktivitätsmetriken
# ---------------------------------------------------------------------------

def test_distinct_user_count_excludes_admin_and_friends(db):
    real_user = _make_user(db, email="real@x.com", hashed_password="x", plan="free")
    admin_user = _make_user(db, email="admin@x.com", hashed_password="x", plan="admin")
    friend_user = _make_user(db, email="friend@x.com", hashed_password="x", plan="friends")

    for u in (real_user, admin_user, friend_user):
        _log_event(db, "analysis_started", u.id, metadata={"mode": "wachstumswerte", "symbol": "AAPL"})

    count = _distinct_user_count(db, "analysis_started")

    assert count == 1


def test_distinct_user_count_includes_pro_and_free_plans(db):
    free_user = _make_user(db, email="free@x.com", hashed_password="x", plan="free")
    pro_user = _make_user(db, email="pro@x.com", hashed_password="x", plan="pro")

    for u in (free_user, pro_user):
        _log_event(db, "user_registered", u.id)

    count = _distinct_user_count(db, "user_registered")

    assert count == 2
