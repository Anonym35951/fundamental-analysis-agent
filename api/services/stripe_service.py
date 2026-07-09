import stripe

from api.core.config import settings


stripe.api_key = settings.STRIPE_SECRET_KEY


def get_pro_price_id(billing_interval: str) -> str:
    if billing_interval == "month":
        return settings.STRIPE_PRICE_ID_PRO_MONTHLY
    if billing_interval == "year":
        return settings.STRIPE_PRICE_ID_PRO_YEARLY

    raise ValueError(f"Invalid billing interval: {billing_interval}")


def create_checkout_session(user_id: int, user_email: str, billing_interval: str) -> str:
    price_id = get_pro_price_id(billing_interval)

    session_params = {
        "mode": "subscription",
        "payment_method_types": ["card"],
        "line_items": [
            {
                "price": price_id,
                "quantity": 1,
            }
        ],
        "success_url": settings.STRIPE_SUCCESS_URL,
        "cancel_url": settings.STRIPE_CANCEL_URL,
        "customer_email": user_email,
        "client_reference_id": str(user_id),
        "metadata": {
            "user_id": str(user_id),
            "plan": "pro",
            "billing_interval": billing_interval,
            "stripe_price_id": price_id,
        },
    }

    # Erst einschalten, wenn Stripe Tax im Dashboard eingerichtet ist
    # (Steuerregistrierung + Ursprungsadresse), sonst schlägt der Checkout fehl.
    if settings.STRIPE_AUTOMATIC_TAX:
        session_params["automatic_tax"] = {"enabled": True}

    session = stripe.checkout.Session.create(**session_params)

    return session.url


def cancel_subscription(stripe_subscription_id: str):
    return stripe.Subscription.modify(
        stripe_subscription_id,
        cancel_at_period_end=True,
    )


def cancel_subscription_immediately(stripe_subscription_id: str):
    """Sofortige Kündigung (nicht zum Periodenende) — wird bei der
    Konto-Löschung genutzt, damit kein Abo auf einen gelöschten Account
    weiterläuft."""
    return stripe.Subscription.cancel(stripe_subscription_id)


def resume_subscription(stripe_subscription_id: str):
    return stripe.Subscription.modify(
        stripe_subscription_id,
        cancel_at_period_end=False,
    )