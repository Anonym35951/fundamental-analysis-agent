import logging
from datetime import datetime
from typing import Literal

import stripe
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.core.rate_limit import limiter

from api.core.config import settings
from api.core.dependencies import get_current_user, get_db
from api.crud.analysis_history import get_usage_summary
from api.models.user import User
from api.services.email_service import (
    send_email_safely,
    send_subscription_canceled_email,
    send_subscription_resumed_email,
)
from api.services.event_service import log_event
from api.services.stripe_service import (
    create_checkout_session,
    cancel_subscription,
    resume_subscription,
)

router = APIRouter(prefix="/billing", tags=["billing"])

logger = logging.getLogger(__name__)

stripe.api_key = settings.STRIPE_SECRET_KEY


class CreateCheckoutSessionRequest(BaseModel):
    billing_interval: Literal["month", "year"]


class CancelSubscriptionRequest(BaseModel):
    # Optional & überspringbar — reine Information für den Betreiber, kein
    # Pflichtfeld, das den Kündigungs-Flow erschwert.
    reason: str | None = Field(default=None, max_length=500)


# =========================
# CHECKOUT SESSION
# =========================
@router.post("/create-checkout-session")
@limiter.limit("5/minute")
def billing_create_checkout_session(
    request: Request,
    payload: CreateCheckoutSessionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not current_user.email:
        raise HTTPException(status_code=400, detail="User email is required")

    checkout_url = create_checkout_session(
        user_id=int(current_user.id),
        user_email=current_user.email,
        billing_interval=payload.billing_interval,
    )

    log_event(
        db,
        "checkout_started",
        user_id=current_user.id,
        metadata={"billing_interval": payload.billing_interval},
    )

    return {"checkout_url": checkout_url}


# =========================
# USAGE SUMMARY (Kündigungs-Flow)
# =========================
@router.get("/usage-summary")
def billing_usage_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Neutraler Nutzungs-Rückblick, den das Frontend im Kündigungs-Dialog
    zeigt (z. B. "Du hast 87 Analysen über 24 Unternehmen ausgeführt") —
    reine Information, kein Guilt-Tripping."""
    return get_usage_summary(db, current_user.id)


# =========================
# CUSTOMER PORTAL (NEU)
# =========================
@router.post("/create-portal-session")
def billing_create_portal_session(
    current_user: User = Depends(get_current_user),
):
    if not current_user.stripe_customer_id:
        raise HTTPException(status_code=400, detail="No Stripe customer found")

    session = stripe.billing_portal.Session.create(
        customer=current_user.stripe_customer_id,
        return_url=f"{settings.FRONTEND_URL}/app/account",
    )

    return {"url": session.url}


# =========================
# CANCEL SUBSCRIPTION
# =========================
@router.post("/cancel-subscription")
def billing_cancel_subscription(
    background_tasks: BackgroundTasks,
    payload: CancelSubscriptionRequest = CancelSubscriptionRequest(),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not current_user.stripe_subscription_id:
        raise HTTPException(status_code=400, detail="No active subscription found")

    cancel_subscription(current_user.stripe_subscription_id)

    subscription = stripe.Subscription.retrieve(current_user.stripe_subscription_id)

    current_period_end_ts = None
    try:
        current_period_end_ts = subscription["current_period_end"]
    except (KeyError, TypeError):
        current_period_end_ts = None

    if current_period_end_ts is None:
        try:
            current_period_end_ts = subscription["items"]["data"][0]["current_period_end"]
        except (KeyError, TypeError, IndexError):
            current_period_end_ts = None

    logger.debug(
        "Cancel subscription: status=%s current_period_end=%s",
        getattr(subscription, "status", None),
        current_period_end_ts,
    )

    cancel_at_period_end = getattr(subscription, "cancel_at_period_end", False)

    current_period_end = None
    if current_period_end_ts is not None:
        current_period_end = datetime.utcfromtimestamp(current_period_end_ts)

    user = db.query(User).filter(User.id == current_user.id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if cancel_at_period_end:
        user.billing_status = "canceling"

    user.current_period_end = current_period_end

    db.commit()
    db.refresh(user)

    formatted_current_period_end = (
        user.current_period_end.strftime("%d.%m.%Y")
        if user.current_period_end is not None
        else None
    )

    background_tasks.add_task(
        send_email_safely,
        send_subscription_canceled_email,
        to_email=user.email,
        current_period_end=formatted_current_period_end,
    )

    logger.info(
        "Billing cancel subscription: queued cancellation email for user_id=%s (current_period_end=%s)",
        user.id,
        formatted_current_period_end,
    )

    log_event(
        db,
        "subscription_cancel_requested",
        user_id=user.id,
        metadata={"reason": payload.reason} if payload.reason else None,
    )

    return {
        "message": "Subscription will be canceled at the end of the current billing period.",
        "cancel_at_period_end": cancel_at_period_end,
        "current_period_end": user.current_period_end,
    }


# =========================
# RESUME SUBSCRIPTION
# =========================
@router.post("/resume-subscription")
def billing_resume_subscription(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not current_user.stripe_subscription_id:
        raise HTTPException(status_code=400, detail="No active subscription found")

    if current_user.billing_status != "canceling":
        raise HTTPException(
            status_code=400,
            detail="Subscription is not scheduled for cancellation",
        )

    resume_subscription(current_user.stripe_subscription_id)

    subscription = stripe.Subscription.retrieve(current_user.stripe_subscription_id)

    current_period_end_ts = None
    try:
        current_period_end_ts = subscription["current_period_end"]
    except (KeyError, TypeError):
        current_period_end_ts = None

    if current_period_end_ts is None:
        try:
            current_period_end_ts = subscription["items"]["data"][0]["current_period_end"]
        except (KeyError, TypeError, IndexError):
            current_period_end_ts = None

    cancel_at_period_end = getattr(subscription, "cancel_at_period_end", False)

    current_period_end = None
    if current_period_end_ts is not None:
        current_period_end = datetime.utcfromtimestamp(current_period_end_ts)

    user = db.query(User).filter(User.id == current_user.id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    user.billing_status = "active"
    user.current_period_end = current_period_end

    db.commit()
    db.refresh(user)

    formatted_current_period_end = (
        user.current_period_end.strftime("%d.%m.%Y")
        if user.current_period_end is not None
        else None
    )

    background_tasks.add_task(
        send_email_safely,
        send_subscription_resumed_email,
        to_email=user.email,
        current_period_end=formatted_current_period_end,
    )

    logger.info(
        "Billing resume subscription: queued resume email for user_id=%s (current_period_end=%s)",
        user.id,
        formatted_current_period_end,
    )

    log_event(db, "subscription_resumed", user_id=user.id)

    return {
        "message": "Subscription will continue normally.",
        "cancel_at_period_end": cancel_at_period_end,
        "current_period_end": user.current_period_end,
    }