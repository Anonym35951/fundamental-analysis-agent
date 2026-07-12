# api/routes/admin_stats.py
"""Aggregierte Kennzahlen fürs private Admin-Analytics-Dashboard (/app/admin
im Frontend). Kein Dritt-Anbieter-Tracking — alle Daten kommen aus
product_events (api/models/product_event.py) und der users-Tabelle."""
from datetime import datetime, timedelta
from api.utils.time import utcnow

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import Date, and_, case, cast, func
from sqlalchemy.orm import Session

from api.core.dependencies import get_db, require_admin
from api.core.rate_limit import limiter
from api.models.product_event import ProductEvent
from api.models.user import User

router = APIRouter(prefix="/admin/stats", tags=["admin-stats"])

# Eigene Accounts (Betreiber-Admin, vorab freigeschaltete "friends"-Tester)
# erzeugen dieselben Events wie echte Nutzer. Bei kleiner Nutzerbasis (Soft
# Launch!) dominiert sonst die Eigennutzung DAU/WAU/MAU, Funnel und die
# Analyse-Charts. Siehe LAUNCH.md P2-16.
INTERNAL_PLANS = ("admin", "friends")

# Pro-Preise für die MRR-Schätzung. Kein Live-Stripe-Call pro Dashboard-
# Aufruf (wäre ein Stripe-API-Call pro Pro-User) - muss manuell synchron
# gehalten werden, falls sich der Preis in Stripe ändert (siehe
# STRIPE_PRICE_ID_PRO_MONTHLY/YEARLY in api/core/config.py). Siehe LAUNCH.md
# P1-7.
PRO_MONTHLY_PRICE_EUR = 50.0
PRO_YEARLY_PRICE_EUR = 500.0
PRO_YEARLY_MONTHLY_EQUIVALENT_EUR = PRO_YEARLY_PRICE_EUR / 12


def _monthly_price_for_interval(billing_interval: str | None) -> float | None:
    """None, wenn das Intervall nie mit Stripe synchronisiert wurde - der
    Aufrufer muss diesen Fall explizit als "unbekannt" auswerten statt ihn
    stillschweigend als monatlich zu werten (vorheriger Bug, LAUNCH.md
    P1-7)."""
    if billing_interval == "year":
        return PRO_YEARLY_MONTHLY_EQUIVALENT_EUR
    if billing_interval == "month":
        return PRO_MONTHLY_PRICE_EUR
    return None


def _internal_user_ids_subquery(db: Session):
    return db.query(User.id).filter(User.plan.in_(INTERNAL_PLANS)).subquery()


def _external_events(db: Session, event_type: str):
    """Basisquery für product_events eines Typs, ohne interne (admin/
    friends) Accounts - siehe INTERNAL_PLANS."""
    internal_ids = _internal_user_ids_subquery(db)
    return db.query(ProductEvent).filter(
        ProductEvent.event_type == event_type,
        ProductEvent.user_id.isnot(None),
        ProductEvent.user_id.notin_(db.query(internal_ids.c.id)),
    )


def _distinct_user_count(db: Session, event_type: str) -> int:
    return (
        _external_events(db, event_type)
        .with_entities(func.count(func.distinct(ProductEvent.user_id)))
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
    Limit-Hit -> Checkout-Start -> zahlender Kunde. Schließt interne
    Accounts aus (siehe INTERNAL_PLANS)."""
    analysis_counts_per_user = (
        _external_events(db, "analysis_started")
        .with_entities(ProductEvent.user_id, func.count(ProductEvent.id).label("cnt"))
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
    jeweiligen Zeitraum (jede Aktion zählt, nicht nur Analysen). Schließt
    interne Accounts aus (siehe INTERNAL_PLANS)."""
    now = utcnow()
    internal_ids = _internal_user_ids_subquery(db)

    def active_users_since(delta: timedelta) -> int:
        since = now - delta
        return (
            db.query(func.count(func.distinct(ProductEvent.user_id)))
            .filter(
                ProductEvent.created_at >= since,
                ProductEvent.user_id.isnot(None),
                ProductEvent.user_id.notin_(db.query(internal_ids.c.id)),
            )
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
    """Zeitreihe für Recharts: Registrierungen und gestartete Analysen pro
    Tag. Schließt interne Accounts aus (siehe INTERNAL_PLANS).

    Bekannte Einschränkung (LAUNCH.md P2-21): Tagesgrenzen laufen auf UTC-
    Kalendertagen (cast auf Date, kein Zeitzonen-Offset), für ein
    DE-Produkt also um 1-2h verschoben (CET/CEST je nach Sommerzeit) -
    Events kurz vor/nach Mitternacht Berlin-Zeit können im falschen
    Tages-Balken landen. Bewusst nicht auf Europe/Berlin umgestellt: die
    korrekte Umrechnung müsste die DST-Umstellung zweimal im Jahr abbilden
    und in Postgres (Prod) vs. SQLite (Tests) unterschiedlich funktionieren
    - ein Trend-Chart mit 1-2h-Rand-Unschärfe ist das kleinere Risiko als
    eine fehlerhafte Zeitzonen-Umrechnung, die sich in den Tests nicht
    zuverlässig nachstellen lässt."""
    since = utcnow() - timedelta(days=days)
    day_col = cast(ProductEvent.created_at, Date)
    internal_ids = _internal_user_ids_subquery(db)

    def daily_counts(event_type: str) -> dict[str, int]:
        rows = (
            db.query(day_col.label("day"), func.count(ProductEvent.id))
            .filter(
                ProductEvent.event_type == event_type,
                ProductEvent.created_at >= since,
                ProductEvent.user_id.isnot(None),
                ProductEvent.user_id.notin_(db.query(internal_ids.c.id)),
            )
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


def _compute_analyses_breakdown(db: Session) -> dict:
    """Analysen pro Modus + meistanalysierte Symbole. Schließt interne
    Accounts aus (siehe INTERNAL_PLANS) - sonst dominieren eigene Testläufe
    die "Top Symbole". Reine Funktion (kein Request/Depends), damit sie ohne
    HTTP-Layer testbar ist - Stil wie _compute_subscription_stats."""
    mode_col = ProductEvent.event_metadata["mode"].as_string()
    source_col = ProductEvent.event_metadata["source"].as_string()
    symbol_col = ProductEvent.event_metadata["symbol"].as_string()
    base_query = _external_events(db, "analysis_started")

    # Compare startet Jobs über denselben "custom"-Endpunkt wie der
    # Custom-Analysis-Builder (siehe custom_analysis.py::_launch_custom_job)
    # - ohne diese Fallunterscheidung waren beide im Modus-Chart nicht
    # unterscheidbar (LAUNCH.md P2-17). Events vor dieser Änderung haben kein
    # "source"-Feld -> source_col ist NULL -> fällt korrekt auf "custom".
    effective_mode_col = case(
        (and_(mode_col == "custom", source_col == "compare"), "compare"),
        else_=mode_col,
    )

    by_mode = (
        base_query
        .with_entities(effective_mode_col.label("mode"), func.count(ProductEvent.id).label("count"))
        .group_by(effective_mode_col)
        .order_by(func.count(ProductEvent.id).desc())
        .all()
    )

    top_symbols = (
        base_query
        .with_entities(symbol_col.label("symbol"), func.count(ProductEvent.id).label("count"))
        .group_by(symbol_col)
        .order_by(func.count(ProductEvent.id).desc())
        .limit(20)
        .all()
    )

    return {
        "by_mode": [{"mode": mode, "count": count} for mode, count in by_mode],
        "top_symbols": [{"symbol": symbol, "count": count} for symbol, count in top_symbols],
    }


@router.get("/analyses")
@limiter.limit("20/minute")
def get_analyses_breakdown(
    request: Request,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    return _compute_analyses_breakdown(db)


def _compute_subscription_stats(db: Session) -> dict:
    """MRR und Abo-Kennzahlen, getrennt nach Billing-Status:
    - active: zählt in mrr_eur (gesunde, wiederkehrende Umsatzbasis).
    - canceling/past_due: zählt in at_risk_mrr_eur (noch bezahlt, aber
      absehbar wegfallend) - NICHT in mrr_eur, sonst wird MRR systematisch
      überzeichnet (vorheriger Bug, LAUNCH.md P1-7).
    - canceled: ausgeschlossen (kein Umsatz mehr zu erwarten; sollte durch
      den downgrade_worker ohnehin auf plan="free" laufen).
    billing_interval == None (nie mit Stripe synchronisiert) fließt in
    keinen Euro-Betrag ein, sondern nur in unknown_billing_interval_count -
    vorher wurde das stillschweigend als monatlich (50 €) gezählt.

    Churn zählt distinct User mit dem `subscription_deleted`-Event der
    letzten 30 Tage (Stripe-Webhook feuert das erst bei der tatsächlichen
    Löschung). Der normale "Kündigung zum Periodenende"-Weg durchläuft vorher
    nur `subscription_status_changed` mit to="canceling" - das ist kein
    abgeschlossener Churn und wird hier bewusst NICHT gezählt (vorheriger
    Bug: Query zählte nur to="canceled", das der normale Weg nie erreicht,
    Churn zeigte daher praktisch immer 0). `subscription_cancel_requested`
    (Klick auf "Kündigen" im Frontend, bevor Stripe bestätigt) wird separat
    als Frühindikator ausgewiesen, nicht als abgeschlossener Churn.

    Reine, DB-Session-parametrisierte Funktion (kein FastAPI-Dependency-
    Aufruf) - direkt testbar ohne Rate-Limiter/Request-Mocking, siehe
    api/tests/test_admin_stats.py."""
    pro_users = (
        db.query(User.billing_status, User.billing_interval)
        .filter(User.plan == "pro")
        .all()
    )

    mrr_eur = 0.0
    monthly_count = 0
    yearly_count = 0
    unknown_interval_count = 0
    at_risk_mrr_eur = 0.0
    at_risk_count = 0

    for billing_status, billing_interval in pro_users:
        price = _monthly_price_for_interval(billing_interval)

        if billing_status == "active":
            if price is None:
                unknown_interval_count += 1
            else:
                mrr_eur += price
                if billing_interval == "year":
                    yearly_count += 1
                else:
                    monthly_count += 1
        elif billing_status in ("canceling", "past_due"):
            at_risk_count += 1
            if price is not None:
                at_risk_mrr_eur += price
            else:
                unknown_interval_count += 1
        # billing_status == "canceled": kein Umsatz, bewusst ignoriert.

    since_30d = utcnow() - timedelta(days=30)

    churned_last_30d = (
        db.query(func.count(func.distinct(ProductEvent.user_id)))
        .filter(
            ProductEvent.event_type == "subscription_deleted",
            ProductEvent.created_at >= since_30d,
            ProductEvent.user_id.isnot(None),
        )
        .scalar()
        or 0
    )

    cancellations_requested_last_30d = (
        db.query(func.count(func.distinct(ProductEvent.user_id)))
        .filter(
            ProductEvent.event_type == "subscription_cancel_requested",
            ProductEvent.created_at >= since_30d,
            ProductEvent.user_id.isnot(None),
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
        "unknown_billing_interval_count": unknown_interval_count,
        "mrr_eur": round(mrr_eur, 2),
        "at_risk_subscriptions": at_risk_count,
        "at_risk_mrr_eur": round(at_risk_mrr_eur, 2),
        "churned_last_30d": churned_last_30d,
        "cancellations_requested_last_30d": cancellations_requested_last_30d,
        "free_users_near_limit": free_users_near_limit,
    }


@router.get("/subscriptions")
@limiter.limit("20/minute")
def get_subscriptions(
    request: Request,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    return _compute_subscription_stats(db)


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
