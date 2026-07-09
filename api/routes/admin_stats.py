# api/routes/admin_stats.py
"""Aggregierte Kennzahlen fürs private Admin-Analytics-Dashboard (/app/admin
im Frontend). Kein Dritt-Anbieter-Tracking — alle Daten kommen aus
product_events (api/models/product_event.py) und der users-Tabelle."""
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import Date, cast, func
from sqlalchemy.orm import Session

from api.core.dependencies import get_db, require_admin
from api.core.rate_limit import limiter
from api.models.product_event import ProductEvent
from api.models.user import User

router = APIRouter(prefix="/admin/stats", tags=["admin-stats"])


def _distinct_user_count(db: Session, event_type: str) -> int:
    return (
        db.query(func.count(func.distinct(ProductEvent.user_id)))
        .filter(ProductEvent.event_type == event_type, ProductEvent.user_id.isnot(None))
        .scalar()
        or 0
    )


@router.get("/funnel")
@limiter.limit("20/minute")
def get_funnel(
    request: Request,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    """Registrierung -> Verifizierung -> 1. Analyse -> 5. Analyse ->
    Limit-Hit -> Checkout-Start -> zahlender Kunde."""
    analysis_counts_per_user = (
        db.query(ProductEvent.user_id, func.count(ProductEvent.id).label("cnt"))
        .filter(
            ProductEvent.event_type == "analysis_started",
            ProductEvent.user_id.isnot(None),
        )
        .group_by(ProductEvent.user_id)
        .all()
    )

    return {
        "registered": _distinct_user_count(db, "user_registered"),
        "email_verified": _distinct_user_count(db, "email_verified"),
        "first_analysis": sum(1 for _, cnt in analysis_counts_per_user if cnt >= 1),
        "five_analyses": sum(1 for _, cnt in analysis_counts_per_user if cnt >= 5),
        "quota_hit": _distinct_user_count(db, "quota_exceeded"),
        "checkout_started": _distinct_user_count(db, "checkout_started"),
        "subscription_started": _distinct_user_count(db, "subscription_started"),
    }


@router.get("/activity")
@limiter.limit("20/minute")
def get_activity(
    request: Request,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    """DAU/WAU/MAU, gemessen an Nutzern mit mindestens einem Event im
    jeweiligen Zeitraum (jede Aktion zählt, nicht nur Analysen)."""
    now = datetime.utcnow()

    def active_users_since(delta: timedelta) -> int:
        since = now - delta
        return (
            db.query(func.count(func.distinct(ProductEvent.user_id)))
            .filter(ProductEvent.created_at >= since, ProductEvent.user_id.isnot(None))
            .scalar()
            or 0
        )

    return {
        "dau": active_users_since(timedelta(days=1)),
        "wau": active_users_since(timedelta(days=7)),
        "mau": active_users_since(timedelta(days=30)),
    }


@router.get("/daily-activity")
@limiter.limit("20/minute")
def get_daily_activity(
    request: Request,
    days: int = Query(default=30, ge=1, le=180),
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    """Zeitreihe für Recharts: Registrierungen und gestartete Analysen pro Tag."""
    since = datetime.utcnow() - timedelta(days=days)
    day_col = cast(ProductEvent.created_at, Date)

    def daily_counts(event_type: str) -> dict[str, int]:
        rows = (
            db.query(day_col.label("day"), func.count(ProductEvent.id))
            .filter(ProductEvent.event_type == event_type, ProductEvent.created_at >= since)
            .group_by(day_col)
            .all()
        )
        return {str(day): count for day, count in rows}

    registrations_by_day = daily_counts("user_registered")
    analyses_by_day = daily_counts("analysis_started")

    all_days = sorted(set(registrations_by_day) | set(analyses_by_day))
    return [
        {
            "date": day,
            "registrations": registrations_by_day.get(day, 0),
            "analyses": analyses_by_day.get(day, 0),
        }
        for day in all_days
    ]


@router.get("/analyses")
@limiter.limit("20/minute")
def get_analyses_breakdown(
    request: Request,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    """Analysen pro Modus + meistanalysierte Symbole."""
    mode_col = ProductEvent.event_metadata["mode"].as_string()
    symbol_col = ProductEvent.event_metadata["symbol"].as_string()

    by_mode = (
        db.query(mode_col.label("mode"), func.count(ProductEvent.id).label("count"))
        .filter(ProductEvent.event_type == "analysis_started")
        .group_by(mode_col)
        .order_by(func.count(ProductEvent.id).desc())
        .all()
    )

    top_symbols = (
        db.query(symbol_col.label("symbol"), func.count(ProductEvent.id).label("count"))
        .filter(ProductEvent.event_type == "analysis_started")
        .group_by(symbol_col)
        .order_by(func.count(ProductEvent.id).desc())
        .limit(20)
        .all()
    )

    return {
        "by_mode": [{"mode": mode, "count": count} for mode, count in by_mode],
        "top_symbols": [{"symbol": symbol, "count": count} for symbol, count in top_symbols],
    }


@router.get("/subscriptions")
@limiter.limit("20/minute")
def get_subscriptions(
    request: Request,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    """Aktive Abos, geschätzte MRR (aus Preis × Intervall, kein Stripe-Call
    nötig), Churn der letzten 30 Tage."""
    pro_users = db.query(User.billing_interval).filter(User.plan == "pro").all()

    mrr = 0.0
    monthly_count = 0
    yearly_count = 0
    for (billing_interval,) in pro_users:
        if billing_interval == "year":
            mrr += 500 / 12
            yearly_count += 1
        else:
            mrr += 50
            monthly_count += 1

    since_30d = datetime.utcnow() - timedelta(days=30)
    churned_30d = (
        db.query(func.count(func.distinct(ProductEvent.user_id)))
        .filter(
            ProductEvent.event_type == "subscription_status_changed",
            ProductEvent.event_metadata["to"].as_string() == "canceled",
            ProductEvent.created_at >= since_30d,
        )
        .scalar()
        or 0
    )

    free_users_near_limit = (
        db.query(func.count(User.id))
        .filter(
            User.plan == "free",
            User.monthly_request_limit.isnot(None),
            User.monthly_request_count >= User.monthly_request_limit * 0.8,
        )
        .scalar()
        or 0
    )

    return {
        "active_pro_subscriptions": monthly_count + yearly_count,
        "monthly_subscriptions": monthly_count,
        "yearly_subscriptions": yearly_count,
        "mrr_eur": round(mrr, 2),
        "churned_last_30d": churned_30d,
        "free_users_near_limit": free_users_near_limit,
    }


@router.get("/near-limit-users")
@limiter.limit("20/minute")
def get_near_limit_users(
    request: Request,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    """Free-User, die kurz vor ihrem Monatslimit stehen (>= 80 %) — Kandidaten
    für gezielte Upgrade-Ansprache."""
    users = (
        db.query(User)
        .filter(
            User.plan == "free",
            User.monthly_request_limit.isnot(None),
            User.monthly_request_count >= User.monthly_request_limit * 0.8,
        )
        .order_by(User.monthly_request_count.desc())
        .all()
    )

    return [
        {
            "email": user.email,
            "monthly_request_count": user.monthly_request_count,
            "monthly_request_limit": user.monthly_request_limit,
        }
        for user in users
    ]


@router.get("/churn-reasons")
@limiter.limit("20/minute")
def get_churn_reasons(
    request: Request,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    """Kündigungsgründe, sofern beim Cancel-Flow erfasst (siehe Punkt 4.6 im
    Plan — die Erfassungs-UI existiert noch nicht, daher aktuell meist leer;
    das Event `subscription_cancel_requested` trägt die Gründe, sobald 4.6
    umgesetzt ist)."""
    reason_col = ProductEvent.event_metadata["reason"].as_string()

    rows = (
        db.query(reason_col.label("reason"), func.count(ProductEvent.id).label("count"))
        .filter(
            ProductEvent.event_type == "subscription_cancel_requested",
            reason_col.isnot(None),
        )
        .group_by(reason_col)
        .order_by(func.count(ProductEvent.id).desc())
        .all()
    )

    return [{"reason": reason, "count": count} for reason, count in rows]
