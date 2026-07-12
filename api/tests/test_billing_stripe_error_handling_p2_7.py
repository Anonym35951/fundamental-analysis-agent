"""Regressionstests für LAUNCH_AUDIT.md P2-7: cancel-/resume-subscription
riefen stripe.Subscription.retrieve() (und die modify()-Aufrufe dahinter)
ohne try/except auf - ein Stripe-Ausfall wurde ein roher 500 statt eines
verständlichen 502. api/routes/billing.py::_modify_and_reload_subscription
buendelt jetzt beide Stripe-Calls hinter einem gemeinsamen Fehler-Handler.

Stil wie test_account_deletion.py: direkte Funktionsaufrufe, Stripe gemockt.
"""
from unittest.mock import MagicMock, patch

import pytest
import stripe
from fastapi import BackgroundTasks, HTTPException
from starlette.requests import Request

from api.models.user import User
from api.routes.billing import (
    _modify_and_reload_subscription,
    billing_cancel_subscription,
    billing_resume_subscription,
)


def _make_user(db, **kwargs) -> User:
    defaults = {
        "email": f"user{id(kwargs)}@example.com",
        "hashed_password": "x",
        "stripe_subscription_id": "sub_test123",
        "stripe_customer_id": "cus_test123",
        "billing_status": "active",
    }
    defaults.update(kwargs)
    user = User(**defaults)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ---------------------------------------------------------------------------
# _modify_and_reload_subscription - der eigentliche Fix
# ---------------------------------------------------------------------------

def test_modify_and_reload_returns_subscription_on_success():
    fake_subscription = MagicMock()
    modify_fn = MagicMock()

    with patch("stripe.Subscription.retrieve", return_value=fake_subscription) as mock_retrieve:
        result = _modify_and_reload_subscription(modify_fn, "sub_123")

    modify_fn.assert_called_once_with("sub_123")
    mock_retrieve.assert_called_once_with("sub_123")
    assert result is fake_subscription


def test_modify_and_reload_raises_502_when_modify_fails():
    def _boom(sub_id):
        raise stripe.error.APIConnectionError("network down")

    with pytest.raises(HTTPException) as exc:
        _modify_and_reload_subscription(_boom, "sub_123")

    assert exc.value.status_code == 502


def test_modify_and_reload_raises_502_when_retrieve_fails():
    modify_fn = MagicMock()

    with patch("stripe.Subscription.retrieve", side_effect=stripe.error.APIConnectionError("network down")):
        with pytest.raises(HTTPException) as exc:
            _modify_and_reload_subscription(modify_fn, "sub_123")

    assert exc.value.status_code == 502
    modify_fn.assert_called_once_with("sub_123")


# ---------------------------------------------------------------------------
# Routen-Verdrahtung: 502 statt unbehandeltem 500
# ---------------------------------------------------------------------------

def test_cancel_subscription_route_returns_502_on_stripe_outage(db):
    user = _make_user(db)

    with patch("api.routes.billing.cancel_subscription", side_effect=stripe.error.APIConnectionError("down")):
        with pytest.raises(HTTPException) as exc:
            billing_cancel_subscription(
                background_tasks=BackgroundTasks(), current_user=user, db=db
            )

    assert exc.value.status_code == 502


def test_resume_subscription_route_returns_502_on_stripe_outage(db):
    user = _make_user(db, billing_status="canceling")

    with patch("api.routes.billing.resume_subscription", side_effect=stripe.error.APIConnectionError("down")):
        with pytest.raises(HTTPException) as exc:
            billing_resume_subscription(
                background_tasks=BackgroundTasks(), current_user=user, db=db
            )

    assert exc.value.status_code == 502
