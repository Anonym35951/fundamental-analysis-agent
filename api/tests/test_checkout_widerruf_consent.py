"""Regressionstest für den Widerrufsrecht-Consent im Stripe-Checkout.

TermsPage.tsx behauptet, dass beim Abo-Abschluss eine ausdrueckliche
Zustimmung eingeholt wird, dass die Vertragsausfuehrung vor Ablauf der
Widerrufsfrist beginnt (§ 356 Abs. 4 BGB) - ohne diese Checkbox im
tatsaechlichen Checkout waere die AGB-Aussage falsch und das
Widerrufsrecht wuerde nicht vorzeitig erloeschen. Dieser Test stellt
sicher, dass create_checkout_session die Checkbox tatsaechlich anfordert.

Stil wie test_billing_stripe_error_handling_p2_7.py: Stripe gemockt, kein
Netzwerkzugriff.
"""
from unittest.mock import MagicMock, patch

from api.services.stripe_service import create_checkout_session


def test_checkout_session_requires_terms_of_service_consent():
    fake_session = MagicMock(url="https://checkout.stripe.com/session/test")

    with patch("stripe.checkout.Session.create", return_value=fake_session) as mock_create:
        url = create_checkout_session(user_id=1, user_email="user@example.com", billing_interval="month")

    assert url == fake_session.url
    mock_create.assert_called_once()
    kwargs = mock_create.call_args.kwargs

    assert kwargs["consent_collection"] == {"terms_of_service": "required"}
    message = kwargs["custom_text"]["terms_of_service_acceptance"]["message"]
    assert "Widerrufsrecht" in message
    assert "vor Ablauf der" in message
    assert "/legal/terms" in message
