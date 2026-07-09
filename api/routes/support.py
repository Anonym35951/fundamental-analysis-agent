from fastapi import APIRouter, BackgroundTasks, Depends, Request

from api.core.dependencies import get_current_user_optional
from api.core.rate_limit import limiter
from api.models.user import User
from api.schemas.support import SupportRequest
from api.services.email_service import send_email_safely, send_support_request_email

router = APIRouter(prefix="/support", tags=["support"])


@router.post("/contact", status_code=202)
@limiter.limit("3/minute")
def submit_support_request(
    request: Request,
    payload: SupportRequest,
    background_tasks: BackgroundTasks,
    current_user: User | None = Depends(get_current_user_optional),
):
    """Oeffentliches Support-Kontaktformular (eingeloggt oder anonym nutzbar,
    siehe frontend/src/components/support/SupportForm.tsx). Sendet immer 202
    - Mail-Fehler werden von send_email_safely geloggt, nie dem Absender
    angezeigt (gleiche Philosophie wie alle anderen Mail-Versandpfade)."""
    user_context = (
        f"Eingeloggt als User #{current_user.id}, Plan: {current_user.plan}"
        if current_user
        else None
    )

    background_tasks.add_task(
        send_email_safely,
        send_support_request_email,
        payload.category,
        payload.message,
        payload.email,
        user_context,
    )

    return {"detail": "Anfrage gesendet."}
