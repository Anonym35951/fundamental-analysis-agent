import logging
from datetime import datetime, timedelta
from api.utils.time import utcnow

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
import stripe

from api.core.config import settings
from api.core.database import SessionLocal
from api.models.user import User
from api.services.email_service import (
    send_email_safely,
    send_payment_failed_email,
    send_payment_method_added_email,
    send_payment_method_updated_email,
    send_subscription_canceled_email,
    send_subscription_plan_changed_email,
    send_subscription_resumed_email,
    send_subscription_started_email,
)
from api.services.event_service import log_event
from api.services.stripe_event_service import (
    is_stripe_event_processed,
    mark_stripe_event_processed,
)
from api.services.user_service import update_user_plan

router = APIRouter(prefix="/stripe", tags=["stripe"])

logger = logging.getLogger(__name__)

stripe.api_key = settings.STRIPE_SECRET_KEY


def _get_metadata_value(metadata, key: str) -> str | None:
    if metadata is None:
        return None

    try:
        return metadata[key]
    except (KeyError, TypeError):
        return None


def _get_user_by_subscription_or_customer(
    db,
    stripe_subscription_id: str | None = None,
    stripe_customer_id: str | None = None,
) -> User | None:
    user = None

    if stripe_subscription_id is not None:
        user = (
            db.query(User)
            .filter(User.stripe_subscription_id == stripe_subscription_id)
            .first()
        )

    if user is None and stripe_customer_id is not None:
        user = (
            db.query(User)
            .filter(User.stripe_customer_id == stripe_customer_id)
            .first()
        )

    return user


def _format_datetime_for_email(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.strftime("%d.%m.%Y")


def _build_payment_method_label(payment_method) -> str | None:
    if payment_method is None:
        return None

    payment_method_type = None
    try:
        payment_method_type = payment_method["type"]
    except (KeyError, TypeError):
        payment_method_type = getattr(payment_method, "type", None)

    if payment_method_type == "card":
        try:
            brand = payment_method["card"]["brand"]
            last4 = payment_method["card"]["last4"]
            return f"Card ({brand.upper()} •••• {last4})"
        except (KeyError, TypeError):
            return "Card"

    if payment_method_type == "paypal":
        return "PayPal"

    if payment_method_type == "sepa_debit":
        try:
            last4 = payment_method["sepa_debit"]["last4"]
            return f"SEPA Direct Debit (•••• {last4})"
        except (KeyError, TypeError):
            return "SEPA Direct Debit"

    if payment_method_type == "us_bank_account":
        return "Bank account"

    if payment_method_type is not None:
        return str(payment_method_type).replace("_", " ").title()

    return None


@router.post("/webhook")
async def stripe_webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    if not sig_header:
        raise HTTPException(status_code=400, detail="Missing Stripe signature header")

    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=sig_header,
            secret=settings.STRIPE_WEBHOOK_SECRET,
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_id = event["id"]
    event_type = event["type"]

    db = SessionLocal()
    try:
        if is_stripe_event_processed(db, event_id):
            return {"received": True, "event_type": event_type, "duplicate": True}

        # =========================
        # CHECKOUT COMPLETED
        # =========================
        if event_type == "checkout.session.completed":
            session = event["data"]["object"]

            client_reference_id = getattr(session, "client_reference_id", None)
            metadata = getattr(session, "metadata", None)

            metadata_user_id = _get_metadata_value(metadata, "user_id")
            metadata_billing_interval = _get_metadata_value(metadata, "billing_interval")
            metadata_stripe_price_id = _get_metadata_value(metadata, "stripe_price_id")

            user_id = client_reference_id or metadata_user_id
            stripe_customer_id = getattr(session, "customer", None)
            stripe_subscription_id = getattr(session, "subscription", None)

            updated_user = None

            if user_id is not None:
                updated_user = update_user_plan(
                    db=db,
                    user_id=int(user_id),
                    new_plan="pro",
                    stripe_customer_id=stripe_customer_id,
                    stripe_subscription_id=stripe_subscription_id,
                    billing_interval=metadata_billing_interval,
                    stripe_price_id=metadata_stripe_price_id,
                )

                if updated_user is not None:
                    updated_user.billing_status = "active"
                    updated_user.grace_until = None
                    db.commit()
                    db.refresh(updated_user)

                    background_tasks.add_task(
                        send_email_safely,
                        send_subscription_started_email,
                        to_email=updated_user.email,
                        billing_interval=updated_user.billing_interval,
                    )

                    log_event(
                        db,
                        "subscription_started",
                        user_id=updated_user.id,
                        metadata={"billing_interval": metadata_billing_interval},
                    )

                logger.info(
                    "Webhook processed checkout.session.completed: %s",
                    {
                        "user_id": user_id,
                        "updated": updated_user is not None,
                        "customer": stripe_customer_id,
                        "subscription": stripe_subscription_id,
                        "billing_interval": metadata_billing_interval,
                        "stripe_price_id": metadata_stripe_price_id,
                    },
                )
            else:
                logger.warning(
                    "Webhook received checkout.session.completed without user_id"
                )

        # =========================
        # SUBSCRIPTION UPDATED
        # =========================
        elif event_type == "customer.subscription.updated":
            subscription = event["data"]["object"]

            stripe_subscription_id = getattr(subscription, "id", None)
            stripe_customer_id = getattr(subscription, "customer", None)
            status = getattr(subscription, "status", None)

            logger.debug(
                "subscription.updated received: subscription=%s status=%s",
                stripe_subscription_id,
                status,
            )

            cancel_at_period_end = getattr(subscription, "cancel_at_period_end", False)
            cancel_at_ts = getattr(subscription, "cancel_at", None)

            has_scheduled_cancellation = (
                cancel_at_period_end or cancel_at_ts is not None
            )

            logger.debug(
                "Cancel debug: %s",
                {
                    "status": status,
                    "cancel_at_period_end": cancel_at_period_end,
                    "cancel_at": cancel_at_ts,
                    "has_scheduled_cancellation": has_scheduled_cancellation,
                },
            )

            current_period_end_ts = None

            try:
                current_period_end_ts = subscription["current_period_end"]
            except (KeyError, TypeError):
                current_period_end_ts = None

            if current_period_end_ts is None:
                try:
                    current_period_end_ts = subscription["items"]["data"][0][
                        "current_period_end"
                    ]
                except (KeyError, TypeError, IndexError):
                    current_period_end_ts = None

            current_period_end = None
            if current_period_end_ts is not None:
                current_period_end = datetime.utcfromtimestamp(current_period_end_ts)

            fresh_subscription = stripe.Subscription.retrieve(stripe_subscription_id)

            price = None
            try:
                price = fresh_subscription["items"]["data"][0]["price"]
            except (KeyError, TypeError, IndexError):
                price = None

            billing_interval = None
            stripe_price_id = None

            if price is not None:
                try:
                    stripe_price_id = price["id"]
                except (KeyError, TypeError):
                    stripe_price_id = None

                try:
                    billing_interval = price["recurring"]["interval"]
                except (KeyError, TypeError):
                    billing_interval = None

                if billing_interval is None:
                    try:
                        billing_interval = fresh_subscription["items"]["data"][0][
                            "plan"
                        ]["interval"]
                    except (KeyError, TypeError, IndexError):
                        billing_interval = None

            user = _get_user_by_subscription_or_customer(
                db=db,
                stripe_subscription_id=stripe_subscription_id,
                stripe_customer_id=stripe_customer_id,
            )

            if user is not None:
                previous_billing_status = user.billing_status
                previous_billing_interval = user.billing_interval
                previous_stripe_price_id = user.stripe_price_id

                if status in ["active", "trialing"]:
                    user.plan = "pro"
                    user.grace_until = None

                    if has_scheduled_cancellation:
                        user.billing_status = "canceling"
                    else:
                        user.billing_status = "active"

                elif status in ["past_due", "unpaid"]:
                    user.billing_status = "past_due"
                    user.grace_until = utcnow() + timedelta(hours=24)

                elif status in ["canceled", "incomplete_expired"]:
                    user.billing_status = "canceled"
                    user.plan = "free"
                    user.grace_until = None
                    user.stripe_subscription_id = None

                billing_status_changed = user.billing_status != previous_billing_status

                user.billing_interval = billing_interval
                user.stripe_price_id = stripe_price_id
                user.current_period_end = current_period_end

                logger.info(
                    "Subscription updated: %s",
                    {
                        "user_id": user.id,
                        "status": status,
                        "previous_billing_status": previous_billing_status,
                        "billing_status": user.billing_status,
                        "previous_billing_interval": previous_billing_interval,
                        "billing_interval": billing_interval,
                        "previous_stripe_price_id": previous_stripe_price_id,
                        "stripe_price_id": stripe_price_id,
                        "current_period_end": current_period_end,
                    },
                )

                db.commit()
                db.refresh(user)

                if billing_status_changed:
                    log_event(
                        db,
                        "subscription_status_changed",
                        user_id=user.id,
                        metadata={
                            "from": previous_billing_status,
                            "to": user.billing_status,
                            "stripe_status": status,
                        },
                    )

                formatted_current_period_end = _format_datetime_for_email(
                    user.current_period_end
                )

                if (
                    user.billing_status == "canceling"
                    and previous_billing_status != "canceling"
                ):
                    background_tasks.add_task(
                        send_email_safely,
                        send_subscription_canceled_email,
                        to_email=user.email,
                        current_period_end=formatted_current_period_end,
                    )

                elif (
                    user.billing_status == "active"
                    and previous_billing_status == "canceling"
                ):
                    background_tasks.add_task(
                        send_email_safely,
                        send_subscription_resumed_email,
                        to_email=user.email,
                        current_period_end=formatted_current_period_end,
                    )

                if (
                    previous_stripe_price_id is not None
                    and stripe_price_id is not None
                    and previous_stripe_price_id != stripe_price_id
                ):
                    background_tasks.add_task(
                        send_email_safely,
                        send_subscription_plan_changed_email,
                        to_email=user.email,
                        old_billing_interval=previous_billing_interval,
                        new_billing_interval=user.billing_interval,
                    )

            else:
                logger.warning(
                    "Webhook received subscription.updated but no user matched: %s",
                    {
                        "customer": stripe_customer_id,
                        "subscription": stripe_subscription_id,
                    },
                )

        # =========================
        # SUBSCRIPTION DELETED
        # =========================
        elif event_type == "customer.subscription.deleted":
            subscription = event["data"]["object"]
            stripe_subscription_id = getattr(subscription, "id", None)
            stripe_customer_id = getattr(subscription, "customer", None)

            updated_user = None
            user = _get_user_by_subscription_or_customer(
                db=db,
                stripe_subscription_id=stripe_subscription_id,
                stripe_customer_id=stripe_customer_id,
            )

            if user is not None:
                updated_user = update_user_plan(
                    db=db,
                    user_id=int(user.id),
                    new_plan="free",
                    stripe_customer_id=user.stripe_customer_id,
                    stripe_subscription_id=None,
                )

                if updated_user is not None:
                    updated_user.billing_status = "canceled"
                    updated_user.grace_until = None
                    updated_user.current_period_end = None
                    db.commit()
                    db.refresh(updated_user)

                    log_event(db, "subscription_deleted", user_id=updated_user.id)

                logger.info(
                    "Webhook processed customer.subscription.deleted: %s",
                    {
                        "user_id": user.id,
                        "updated": updated_user is not None,
                        "customer": stripe_customer_id,
                        "subscription": stripe_subscription_id,
                    },
                )
            else:
                logger.warning(
                    "Webhook received customer.subscription.deleted but no user matched: %s",
                    {
                        "customer": stripe_customer_id,
                        "subscription": stripe_subscription_id,
                    },
                )

        # =========================
        # PAYMENT METHOD ATTACHED
        # =========================
        elif event_type == "payment_method.attached":
            payment_method = event["data"]["object"]
            stripe_customer_id = None

            try:
                stripe_customer_id = payment_method["customer"]
            except (KeyError, TypeError):
                stripe_customer_id = getattr(payment_method, "customer", None)

            user = _get_user_by_subscription_or_customer(
                db=db,
                stripe_customer_id=stripe_customer_id,
            )

            if user is not None:
                payment_method_label = _build_payment_method_label(payment_method)

                background_tasks.add_task(
                    send_email_safely,
                    send_payment_method_added_email,
                    to_email=user.email,
                    payment_method_label=payment_method_label,
                )

                logger.info(
                    "Webhook processed payment_method.attached: %s",
                    {
                        "user_id": user.id,
                        "customer": stripe_customer_id,
                        "payment_method_label": payment_method_label,
                    },
                )
            else:
                logger.warning(
                    "Webhook received payment_method.attached but no user matched: %s",
                    {
                        "customer": stripe_customer_id,
                    },
                )

        # =========================
        # CUSTOMER UPDATED
        # =========================
        elif event_type == "customer.updated":
            customer = event["data"]["object"]
            data = event["data"]

            previous_attributes = {}
            try:
                previous_attributes = data["previous_attributes"]
            except (KeyError, TypeError):
                previous_attributes = {}

            stripe_customer_id = getattr(customer, "id", None)

            user = _get_user_by_subscription_or_customer(
                db=db,
                stripe_customer_id=stripe_customer_id,
            )

            if user is not None:
                old_default_payment_method = None
                new_default_payment_method = None

                try:
                    old_default_payment_method = previous_attributes["invoice_settings"][
                        "default_payment_method"
                    ]
                except (KeyError, TypeError):
                    old_default_payment_method = None

                try:
                    new_default_payment_method = customer["invoice_settings"][
                        "default_payment_method"
                    ]
                except (KeyError, TypeError):
                    new_default_payment_method = None

                payment_method_label = None

                if (
                    new_default_payment_method is not None
                    and old_default_payment_method != new_default_payment_method
                ):
                    payment_method = stripe.PaymentMethod.retrieve(
                        new_default_payment_method
                    )
                    payment_method_label = _build_payment_method_label(payment_method)

                    background_tasks.add_task(
                        send_email_safely,
                        send_payment_method_updated_email,
                        to_email=user.email,
                        payment_method_label=payment_method_label,
                    )

                logger.info(
                    "Webhook processed customer.updated: %s",
                    {
                        "user_id": user.id,
                        "customer": stripe_customer_id,
                        "old_default_payment_method": old_default_payment_method,
                        "new_default_payment_method": new_default_payment_method,
                        "payment_method_label": payment_method_label,
                    },
                )
            else:
                logger.warning(
                    "Webhook received customer.updated but no user matched: %s",
                    {
                        "customer": stripe_customer_id,
                    },
                )

        # =========================
        # PAYMENT FAILED
        # =========================
        elif event_type == "invoice.payment_failed":
            invoice = event["data"]["object"]
            stripe_subscription_id = getattr(invoice, "subscription", None)
            stripe_customer_id = getattr(invoice, "customer", None)

            user = _get_user_by_subscription_or_customer(
                db=db,
                stripe_subscription_id=stripe_subscription_id,
                stripe_customer_id=stripe_customer_id,
            )

            if user is not None:
                user.billing_status = "past_due"
                user.grace_until = utcnow() + timedelta(hours=24)

                db.commit()
                db.refresh(user)

                background_tasks.add_task(send_email_safely, send_payment_failed_email, user.email)

                log_event(db, "payment_failed", user_id=user.id, metadata={"source": "invoice"})

                logger.info(
                    "Webhook processed invoice.payment_failed (soft): %s",
                    {
                        "user_id": user.id,
                        "customer": stripe_customer_id,
                        "subscription": stripe_subscription_id,
                        "billing_status": user.billing_status,
                        "grace_until": user.grace_until,
                    },
                )
            else:
                logger.warning(
                    "Webhook received invoice.payment_failed but no user matched: %s",
                    {
                        "customer": stripe_customer_id,
                        "subscription": stripe_subscription_id,
                    },
                )

        elif event_type == "payment_intent.payment_failed":
            payment_intent = event["data"]["object"]
            stripe_customer_id = getattr(payment_intent, "customer", None)

            user = _get_user_by_subscription_or_customer(
                db=db,
                stripe_customer_id=stripe_customer_id,
            )

            if user is not None:
                user.billing_status = "past_due"
                user.grace_until = utcnow() + timedelta(hours=24)

                db.commit()
                db.refresh(user)

                background_tasks.add_task(send_email_safely, send_payment_failed_email, user.email)

                log_event(db, "payment_failed", user_id=user.id, metadata={"source": "payment_intent"})

                logger.info(
                    "Webhook processed payment_intent.payment_failed: %s",
                    {
                        "user_id": user.id,
                        "customer": stripe_customer_id,
                        "billing_status": user.billing_status,
                        "grace_until": user.grace_until,
                    },
                )
            else:
                logger.warning(
                    "Webhook received payment_intent.payment_failed but no user matched: %s",
                    {
                        "customer": stripe_customer_id,
                    },
                )

        mark_stripe_event_processed(db, event_id, event_type)

    finally:
        db.close()

    return {"received": True, "event_type": event_type}