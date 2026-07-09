import html
import logging
from typing import Callable, Literal

import requests

from api.core.config import settings

logger = logging.getLogger(__name__)

_RESEND_API_URL = "https://api.resend.com/emails"


def send_email_safely(send_fn: Callable[..., bool], *args, **kwargs) -> None:
    """Wrapper for use with FastAPI BackgroundTasks: `send_fn` (one of the
    send_*_email functions below) runs after the response has already been
    sent, so its own send errors are already caught and returned as False —
    this is only the outer safety net for anything unexpected (e.g. a bug
    while building the HTML), which would otherwise surface as an unhandled
    exception in the ASGI background-task machinery instead of a normal
    request-time failure."""
    try:
        send_fn(*args, **kwargs)
    except Exception:
        logger.exception("Unexpected error while sending email in background task")


def _send_email(
    to_email: str,
    subject: str,
    text_body: str,
    html_body: str,
    reply_to: str | None = None,
) -> bool:
    """Sends via the Resend HTTP API (port 443) instead of raw SMTP. Gmail
    SMTP (port 587) was unreliable in production - on Render it failed
    outright with OSError: [Errno 101] Network is unreachable (broken IPv6
    egress for that specific host/port), and even when reachable, mail
    sent from a personal Gmail address (no SPF/DKIM on our own domain) is
    exactly the kind of transactional mail spam filters distrust. An HTTPS
    API call sidesteps the network issue entirely and Resend handles
    proper authentication for the sending domain."""
    payload = {
        "from": settings.EMAIL_FROM,
        "to": [to_email],
        "subject": subject,
        "text": text_body,
        "html": html_body,
    }
    if reply_to:
        payload["reply_to"] = reply_to

    try:
        response = requests.post(
            _RESEND_API_URL,
            headers={"Authorization": f"Bearer {settings.RESEND_API_KEY}"},
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        return True
    except Exception:
        logger.exception("Failed to send email to %s", to_email)
        return False


# ===== Shared HTML building blocks =====
# Chrome/neutral palette mirroring the app's monochrome look (frontend/src/index.css
# dark tokens), adapted for email: light card-on-light-gray body stays for client
# compatibility and spam-score reasons, only the accent colors moved from blue to
# chrome-neutral. "danger" accent keeps the red header for cancellation/failure mails.

EmailAccent = Literal["neutral", "danger"]
InfoBoxTone = Literal["neutral", "success", "danger"]


def _render_info_box(text_html: str, tone: InfoBoxTone = "neutral") -> str:
    palette = {
        "neutral": {"background": "#f4f4f5", "border": "#d4d4d8", "text": "#3f3f46"},
        "success": {"background": "#ecfdf5", "border": "#a7f3d0", "text": "#065f46"},
        "danger": {"background": "#fef2f2", "border": "#fecaca", "text": "#991b1b"},
    }[tone]

    return f"""
              <div style="padding:18px 20px; background-color:{palette["background"]}; border-radius:12px; border:1px solid {palette["border"]}; margin:24px 0;">
                <p style="margin:0; font-size:14px; color:{palette["text"]};">
                  {text_html}
                </p>
              </div>"""


def _render_cta_button(label: str, href: str) -> str:
    return f"""
              <table cellpadding="0" cellspacing="0" style="margin:24px 0;">
                <tr>
                  <td style="background-color:#18181b; border-radius:12px;">
                    <a href="{href}"
                       style="display:inline-block; padding:14px 22px; font-size:15px; font-weight:700; color:#ffffff; text-decoration:none;">
                      {label}
                    </a>
                  </td>
                </tr>
              </table>"""


def _render_email_shell(
    eyebrow: str,
    headline: str,
    subheadline: str,
    body_html: str,
    accent: EmailAccent = "neutral",
) -> str:
    if accent == "danger":
        gradient = "linear-gradient(135deg, #18181b, #7f1d1d)"
        eyebrow_color = "#fecaca"
        subheadline_color = "#fee2e2"
    else:
        gradient = "linear-gradient(135deg, #18181b, #3f3f46)"
        eyebrow_color = "#d4d4d8"
        subheadline_color = "#e4e4e7"

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
</head>
<body style="margin:0; padding:0; background-color:#f5f7fb; font-family:Arial, Helvetica, sans-serif; color:#111827;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f5f7fb; padding:32px 16px;">
    <tr>
      <td align="center">
        <table width="100%" cellpadding="0" cellspacing="0" style="max-width:600px; background-color:#ffffff; border-radius:16px; overflow:hidden; box-shadow:0 8px 30px rgba(0,0,0,0.08);">

          <tr>
            <td style="background:{gradient}; padding:32px 40px;">
              <div style="font-size:14px; letter-spacing:0.08em; text-transform:uppercase; color:{eyebrow_color}; font-weight:700; margin-bottom:10px;">
                {eyebrow}
              </div>
              <h1 style="margin:0; font-size:26px; color:#ffffff;">
                {headline}
              </h1>
              <p style="margin:12px 0 0 0; font-size:16px; color:{subheadline_color};">
                {subheadline}
              </p>
            </td>
          </tr>

          <tr>
            <td style="padding:32px 40px;">
{body_html}

              <p style="margin-top:24px; font-size:15px; color:#52525b;">
                Best regards<br />
                <strong>ComAnalysis</strong>
              </p>
            </td>
          </tr>

          <tr>
            <td style="padding:20px 40px; border-top:1px solid #e5e7eb; background-color:#f9fafb;">
              <p style="margin:0; font-size:12px; color:#6b7280;">
                This is an automated email. Please do not reply.
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>
""".strip()


def send_payment_failed_email(to_email: str) -> bool:
    subject = "Payment failed – please update your payment method"

    billing_url = getattr(
        settings,
        "BILLING_PORTAL_URL",
        "https://fundamental-analysis-agent-9.onrender.com/billing",
    )

    text_body = f"""
Hi,

your recent payment failed.

You still have access for 24 hours.
Please update your payment method to avoid interruption.

Update here: {billing_url}

Best regards
""".strip()

    body_html = f"""
              <p style="font-size:16px; line-height:1.7; color:#27272a;">
                Your <strong>Pro access</strong> remains active for <strong>24 hours</strong>.
                Please update your payment method to avoid interruption.
              </p>
{_render_cta_button("Update payment method", billing_url)}
{_render_info_box("If no update is made, your account will be downgraded automatically after the grace period.", "neutral")}"""

    html_body = _render_email_shell(
        eyebrow="ComAnalysis",
        headline="Payment failed",
        subheadline="Your recent payment could not be processed.",
        body_html=body_html,
        accent="danger",
    )

    return _send_email(
        to_email=to_email,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
    )


def send_password_changed_email(to_email: str) -> bool:
    subject = "Your password has been changed"

    text_body = """
Hi,

this is a confirmation that your password for your ComAnalysis account has been changed successfully.

If you made this change, no further action is required.

If you did not change your password, please secure your account immediately and contact support as soon as possible.

Best regards
ComAnalysis
""".strip()

    body_html = f"""
              <p style="font-size:16px; line-height:1.7; color:#27272a;">
                Your password has been <strong>changed successfully</strong>.
              </p>
{_render_info_box("If this change was made by you, no further action is required.", "success")}
{_render_info_box("If you did not change your password, please secure your account immediately and contact support as soon as possible.", "danger")}"""

    html_body = _render_email_shell(
        eyebrow="ComAnalysis",
        headline="Password changed successfully",
        subheadline="This is a confirmation for your account security.",
        body_html=body_html,
    )

    return _send_email(
        to_email=to_email,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
    )


def send_email_verification_email(to_email: str, verify_link: str) -> bool:
    subject = "Confirm your ComAnalysis email address"

    text_body = f"""
Hi,

welcome to ComAnalysis! Please confirm your email address to activate your account.

Confirm your email here: {verify_link}

This link is valid for 24 hours. If you did not create a ComAnalysis account, you can safely ignore this email.

Best regards
ComAnalysis
""".strip()

    body_html = f"""
              <p style="font-size:16px; line-height:1.7; color:#27272a;">
                Click the button below to confirm your email address. This link is valid for <strong>24 hours</strong>.
              </p>
{_render_cta_button("Confirm email", verify_link)}
{_render_info_box("If you did not create a ComAnalysis account, you can safely ignore this email.", "neutral")}"""

    html_body = _render_email_shell(
        eyebrow="ComAnalysis",
        headline="Confirm your email",
        subheadline="One quick step to activate your account.",
        body_html=body_html,
    )

    return _send_email(
        to_email=to_email,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
    )


def send_password_reset_email(to_email: str, reset_link: str) -> bool:
    subject = "Reset your ComAnalysis password"

    text_body = f"""
Hi,

we received a request to reset your ComAnalysis password.

Reset your password here: {reset_link}

This link is valid for 30 minutes. If you did not request a password reset, you can safely ignore this email.

Best regards
ComAnalysis
""".strip()

    body_html = f"""
              <p style="font-size:16px; line-height:1.7; color:#27272a;">
                Click the button below to choose a new password. This link is valid for <strong>30 minutes</strong>.
              </p>
{_render_cta_button("Reset password", reset_link)}
{_render_info_box("If you did not request a password reset, you can safely ignore this email.", "neutral")}"""

    html_body = _render_email_shell(
        eyebrow="ComAnalysis",
        headline="Reset your password",
        subheadline="We received a request to reset your password.",
        body_html=body_html,
    )

    return _send_email(
        to_email=to_email,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
    )


def send_subscription_started_email(
    to_email: str,
    billing_interval: str | None = None,
) -> bool:
    subject = "Your Pro subscription is now active"

    billing_url = getattr(
        settings,
        "BILLING_PORTAL_URL",
        "https://fundamental-analysis-agent-9.onrender.com/billing",
    )

    interval_label = "monthly"
    if billing_interval == "year":
        interval_label = "yearly"

    text_body = f"""
Hi,

your Pro subscription has been activated successfully.

Your current billing interval: {interval_label}.

You can manage your subscription anytime here:
{billing_url}

Best regards
ComAnalysis
""".strip()

    body_html = f"""
              <p style="font-size:16px; line-height:1.7; color:#27272a;">
                Thank you — your <strong>Pro subscription</strong> has been activated successfully.
              </p>
{_render_info_box(f"Current billing interval: <strong>{interval_label}</strong>", "neutral")}
{_render_cta_button("Manage subscription", billing_url)}"""

    html_body = _render_email_shell(
        eyebrow="ComAnalysis",
        headline="Pro subscription activated",
        subheadline="Your subscription is now active.",
        body_html=body_html,
    )

    return _send_email(
        to_email=to_email,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
    )


def send_subscription_canceled_email(
    to_email: str,
    current_period_end: str | None = None,
) -> bool:
    subject = "Your Pro subscription has been canceled"

    billing_url = getattr(
        settings,
        "BILLING_PORTAL_URL",
        "https://fundamental-analysis-agent-9.onrender.com/billing",
    )

    end_text = (
        f"Your Pro access remains active until {current_period_end}."
        if current_period_end
        else "Your Pro access remains active until the end of the current billing period."
    )

    text_body = f"""
Hi,

your Pro subscription has been canceled successfully.

{end_text}

If you change your mind, you can manage your subscription here:
{billing_url}

Best regards
ComAnalysis
""".strip()

    body_html = f"""
              <p style="font-size:16px; line-height:1.7; color:#27272a;">
                Your <strong>Pro subscription</strong> has been canceled successfully.
              </p>
{_render_info_box(end_text, "danger")}
{_render_cta_button("Manage subscription", billing_url)}"""

    html_body = _render_email_shell(
        eyebrow="ComAnalysis",
        headline="Subscription canceled",
        subheadline="Your cancellation has been scheduled successfully.",
        body_html=body_html,
        accent="danger",
    )

    return _send_email(
        to_email=to_email,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
    )


def send_subscription_resumed_email(
    to_email: str,
    current_period_end: str | None = None,
) -> bool:
    subject = "Your Pro subscription has been resumed"

    billing_url = getattr(
        settings,
        "BILLING_PORTAL_URL",
        "https://fundamental-analysis-agent-9.onrender.com/billing",
    )

    end_text = (
        f"Your current billing period remains active until {current_period_end}."
        if current_period_end
        else "Your current billing period remains active and your subscription will continue normally."
    )

    text_body = f"""
Hi,

your Pro subscription has been resumed successfully.

{end_text}

You can manage your subscription here:
{billing_url}

Best regards
ComAnalysis
""".strip()

    body_html = f"""
              <p style="font-size:16px; line-height:1.7; color:#27272a;">
                Your <strong>Pro subscription</strong> has been resumed successfully.
              </p>
{_render_info_box(end_text, "success")}
{_render_cta_button("Manage subscription", billing_url)}"""

    html_body = _render_email_shell(
        eyebrow="ComAnalysis",
        headline="Subscription resumed",
        subheadline="Your Pro subscription will continue normally.",
        body_html=body_html,
    )

    return _send_email(
        to_email=to_email,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
    )


def send_payment_method_added_email(
    to_email: str,
    payment_method_label: str | None = None,
) -> bool:
    subject = "A new payment method was added to your account"

    billing_url = getattr(
        settings,
        "BILLING_PORTAL_URL",
        "https://fundamental-analysis-agent-9.onrender.com/billing",
    )

    method_text = (
        f"Added payment method: {payment_method_label}."
        if payment_method_label
        else "A new payment method was added to your account."
    )

    text_body = f"""
Hi,

a new payment method has been added successfully to your account.

{method_text}

You can review your billing settings here:
{billing_url}

If this was not you, please review your account immediately.

Best regards
ComAnalysis
""".strip()

    body_html = f"""
              <p style="font-size:16px; line-height:1.7; color:#27272a;">
                A new <strong>payment method</strong> has been added successfully to your account.
              </p>
{_render_info_box(method_text, "neutral")}
{_render_cta_button("Review billing settings", billing_url)}
{_render_info_box("If you did not make this change, please review your account immediately.", "danger")}"""

    html_body = _render_email_shell(
        eyebrow="ComAnalysis",
        headline="Payment method added",
        subheadline="Your billing settings have been updated.",
        body_html=body_html,
    )

    return _send_email(
        to_email=to_email,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
    )


def send_payment_method_updated_email(
    to_email: str,
    payment_method_label: str | None = None,
) -> bool:
    subject = "Your default payment method has been updated"

    billing_url = getattr(
        settings,
        "BILLING_PORTAL_URL",
        "https://fundamental-analysis-agent-9.onrender.com/billing",
    )

    method_text = (
        f"New default payment method: {payment_method_label}."
        if payment_method_label
        else "Your default payment method has been updated successfully."
    )

    text_body = f"""
Hi,

your default payment method has been updated successfully.

{method_text}

You can review your billing settings here:
{billing_url}

If this was not you, please review your account immediately.

Best regards
ComAnalysis
""".strip()

    body_html = f"""
              <p style="font-size:16px; line-height:1.7; color:#27272a;">
                Your <strong>default payment method</strong> has been updated successfully.
              </p>
{_render_info_box(method_text, "neutral")}
{_render_cta_button("Review billing settings", billing_url)}
{_render_info_box("If you did not make this change, please review your account immediately.", "danger")}"""

    html_body = _render_email_shell(
        eyebrow="ComAnalysis",
        headline="Payment method updated",
        subheadline="Your default payment method has been changed.",
        body_html=body_html,
    )

    return _send_email(
        to_email=to_email,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
    )


def send_support_request_email(
    category: str,
    message: str,
    submitter_email: str,
    user_context: str | None = None,
) -> bool:
    """Support-Kontaktformular (öffentliche Kontakt-Seite + /app/support) ->
    Mail an den Betreiber (settings.SUPPORT_EMAIL), Reply-To auf die
    Absender-Adresse gesetzt, damit eine direkte Antwort den Nutzer erreicht."""
    subject = f"[Support] {category} — {submitter_email}"

    text_body = f"""
Neue Support-Anfrage über ComAnalysis.

Kategorie: {category}
Absender: {submitter_email}
{f"Kontext: {user_context}" if user_context else ""}

Nachricht:
{message}
""".strip()

    message_html = html.escape(message).replace("\n", "<br>")

    body_html = f"""
              <p style="font-size:16px; line-height:1.7; color:#27272a;">
                Neue Support-Anfrage über das Kontaktformular.
              </p>
{_render_info_box(f"Kategorie: <strong>{html.escape(category)}</strong><br>Absender: <strong>{html.escape(submitter_email)}</strong>", "neutral")}
{_render_info_box(message_html, "neutral") if message_html else ""}
{_render_info_box(html.escape(user_context), "neutral") if user_context else ""}"""

    html_body = _render_email_shell(
        eyebrow="ComAnalysis",
        headline="Neue Support-Anfrage",
        subheadline=f"Kategorie: {category}",
        body_html=body_html,
    )

    return _send_email(
        to_email=settings.SUPPORT_EMAIL,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
        reply_to=submitter_email,
    )


def _filing_form_label(form: str) -> str:
    return "Jahresbericht (10-K)" if form == "10-K" else "Quartalsbericht (10-Q)"


def send_weekly_watchlist_digest_email(
    to_email: str,
    filing_entries: list[dict],
) -> bool:
    """Wöchentlicher Rückblick für Favoriten mit neuen SEC-Meldungen dieser
    Woche — reine Fakten-Zusammenfassung (ergänzt die sofortige Einzel-
    Benachrichtigung aus send_new_filing_alert_email, siehe
    api/services/watchlist_digest_service.py). 100% Information, null
    Beratung."""
    symbol_count = len(filing_entries)
    subject = (
        f"Dein Wochenrückblick: {symbol_count} neue Meldung"
        + ("en" if symbol_count != 1 else "")
        + " bei deinen Favoriten"
    )

    lines = [
        f"- {entry['symbol']}: {_filing_form_label(entry['form'])}, Stand {entry['filing_date']}"
        for entry in filing_entries
    ]

    text_body = f"""
Hi,

diese Woche gab es bei deinen favorisierten Unternehmen folgende neue SEC-Meldungen:

{chr(10).join(lines)}

Führe deine gespeicherten Analysen mit den aktuellen Daten erneut aus:
{settings.FRONTEND_URL}/app/dashboard

Dies ist eine automatische Zusammenfassung zu deinen Favoriten — keine Anlageempfehlung.

Best regards
ComAnalysis
""".strip()

    rows_html = "".join(
        f"""
              <div style="padding:10px 0; border-bottom:1px solid #e5e7eb;">
                <strong style="color:#18181b;">{entry['symbol']}</strong>
                <span style="color:#52525b;"> — {_filing_form_label(entry['form'])}, Stand {entry['filing_date']}</span>
              </div>"""
        for entry in filing_entries
    )

    body_html = f"""
              <p style="font-size:16px; line-height:1.7; color:#27272a;">
                Diese Woche gab es bei deinen favorisierten Unternehmen folgende neue SEC-Meldungen:
              </p>
              <div style="margin:20px 0;">{rows_html}
              </div>
{_render_cta_button("Zum Dashboard", f"{settings.FRONTEND_URL}/app/dashboard")}
{_render_info_box("Automatische Zusammenfassung zu deinen Favoriten — keine Anlageempfehlung.", "neutral")}"""

    html_body = _render_email_shell(
        eyebrow="ComAnalysis",
        headline="Dein Wochenrückblick",
        subheadline="Neue SEC-Meldungen bei deinen Favoriten dieser Woche.",
        body_html=body_html,
    )

    return _send_email(
        to_email=to_email,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
    )


def send_new_filing_alert_email(
    to_email: str,
    symbol: str,
    form: str,
    filing_date: str,
) -> bool:
    """Rein informative Benachrichtigung für favorisierte Symbole mit neuer
    SEC-Meldung — 100% Information, null Beratung (siehe
    api/services/filing_alert_service.py)."""
    form_label = "Jahresbericht (10-K)" if form == "10-K" else "Quartalsbericht (10-Q)"
    subject = f"{symbol}: neuer {form_label} verfügbar"

    analyze_url = f"{settings.FRONTEND_URL}/app/analyze?symbol={symbol}"

    text_body = f"""
Hi,

{symbol} hat einen neuen {form_label} bei der SEC eingereicht (Stand {filing_date}).

Deine gespeicherte Analyse basiert noch auf den vorherigen Daten. Führe sie mit den aktuellen Zahlen erneut aus:
{analyze_url}

Dies ist eine automatische Information zu deinem Favoriten — keine Anlageempfehlung.

Best regards
ComAnalysis
""".strip()

    body_html = f"""
              <p style="font-size:16px; line-height:1.7; color:#27272a;">
                <strong>{symbol}</strong> hat einen neuen <strong>{form_label}</strong> bei der SEC eingereicht (Stand {filing_date}).
              </p>
              <p style="font-size:16px; line-height:1.7; color:#27272a;">
                Deine gespeicherte Analyse basiert noch auf den vorherigen Daten.
              </p>
{_render_cta_button(f"{symbol}-Analyse aktualisieren", analyze_url)}
{_render_info_box("Automatische Information zu deinem Favoriten — keine Anlageempfehlung.", "neutral")}"""

    html_body = _render_email_shell(
        eyebrow="ComAnalysis",
        headline=f"Neuer {form_label}: {symbol}",
        subheadline="Deine gespeicherte Analyse lässt sich jetzt mit aktuellen Daten neu ausführen.",
        body_html=body_html,
    )

    return _send_email(
        to_email=to_email,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
    )


def send_subscription_plan_changed_email(
    to_email: str,
    old_billing_interval: str | None = None,
    new_billing_interval: str | None = None,
) -> bool:
    subject = "Your subscription plan has been updated"

    billing_url = getattr(
        settings,
        "BILLING_PORTAL_URL",
        "https://fundamental-analysis-agent-9.onrender.com/billing",
    )

    def format_interval(interval: str | None) -> str:
        if interval == "year":
            return "yearly"
        if interval == "month":
            return "monthly"
        return "updated"

    old_label = format_interval(old_billing_interval)
    new_label = format_interval(new_billing_interval)

    change_text = f"Your plan was changed from {old_label} to {new_label}."

    text_body = f"""
Hi,

your subscription plan has been updated successfully.

{change_text}

You can review your subscription here:
{billing_url}

Best regards
ComAnalysis
""".strip()

    body_html = f"""
              <p style="font-size:16px; line-height:1.7; color:#27272a;">
                Your <strong>subscription plan</strong> has been updated successfully.
              </p>
{_render_info_box(change_text, "neutral")}
{_render_cta_button("Review subscription", billing_url)}"""

    html_body = _render_email_shell(
        eyebrow="ComAnalysis",
        headline="Subscription updated",
        subheadline="Your subscription settings have been changed successfully.",
        body_html=body_html,
    )

    return _send_email(
        to_email=to_email,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
    )
