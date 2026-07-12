# api/routes/admin_customers.py
"""Leichtgewichtiges internes CRM fuers private Admin-Dashboard: Kunden-
Suche/-Details, Notizen und Aktivitaets-Timeline. Kein Sales-Pipeline/Lead-
Stage-System - Notizen decken Freitext-Kontext ab, passend zur Projektgroesse.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import or_
from sqlalchemy.orm import Session

from api.core.dependencies import get_db, require_admin
from api.core.rate_limit import limiter
from api.crud.customer_note import create_note, get_notes_for_user
from api.models.product_event import ProductEvent
from api.models.user import User
from api.schemas.customer import (
    CustomerDetail,
    CustomerListItem,
    CustomerNoteCreate,
    CustomerNoteResponse,
    UpdateCustomerPlanRequest,
)
from api.services.event_service import log_event
from api.services.user_service import (
    StripeCancellationError,
    delete_user_account,
    reset_monthly_request_count,
    update_user_plan,
)

router = APIRouter(prefix="/admin/customers", tags=["admin-customers"])


@router.get("", response_model=list[CustomerListItem])
@limiter.limit("20/minute")
def list_customers(
    request: Request,
    search: str | None = Query(default=None),
    plan: str | None = Query(default=None),
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    query = db.query(User)

    if search:
        pattern = f"%{search}%"
        query = query.filter(
            or_(
                User.email.ilike(pattern),
                User.username.ilike(pattern),
                User.first_name.ilike(pattern),
                User.last_name.ilike(pattern),
            )
        )

    if plan:
        query = query.filter(User.plan == plan)

    return query.order_by(User.created_at.desc()).limit(200).all()


@router.get("/{user_id}", response_model=CustomerDetail)
@limiter.limit("20/minute")
def get_customer(
    request: Request,
    user_id: int,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin),
):
    customer = db.query(User).filter(User.id == user_id).first()
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    # Audit-Log für CRM-Zugriffe (LAUNCH.md P2-21) - wer hat wann in wessen
    # Kundendaten reingeschaut. Nur der Zugriffszeitpunkt/die Akteure, keine
    # Kopie der eingesehenen Daten selbst.
    log_event(
        db,
        "admin_customer_viewed",
        user_id=current_admin.id,
        metadata={"viewed_user_id": user_id},
    )
    return customer


@router.get("/{user_id}/notes", response_model=list[CustomerNoteResponse])
@limiter.limit("20/minute")
def list_customer_notes(
    request: Request,
    user_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    return get_notes_for_user(db, user_id)


@router.post("/{user_id}/notes", response_model=CustomerNoteResponse)
@limiter.limit("20/minute")
def add_customer_note(
    request: Request,
    user_id: int,
    data: CustomerNoteCreate,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin),
):
    customer = db.query(User).filter(User.id == user_id).first()
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return create_note(db, user_id=user_id, admin_author_id=current_admin.id, note_text=data.note)


@router.get("/{user_id}/activity")
@limiter.limit("20/minute")
def get_customer_activity(
    request: Request,
    user_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    """Read-only Sicht auf die bestehende product_events-Tabelle, gescoped
    auf einen Nutzer - kein Duplikat-Logging, reine Anzeige."""
    events = (
        db.query(ProductEvent)
        .filter(ProductEvent.user_id == user_id)
        .order_by(ProductEvent.created_at.desc())
        .limit(100)
        .all()
    )
    return [
        {
            "id": event.id,
            "event_type": event.event_type,
            "event_metadata": event.event_metadata,
            "created_at": event.created_at,
        }
        for event in events
    ]


# Plans, die ein Admin ueber das CRM vergeben darf. "admin" bewusst
# ausgeschlossen (Defense-in-Depth, zusaetzlich zum Frontend-Dropdown ohne
# diese Option) - Admin-Rechte bleiben exklusiv ueber scripts/set_admin.py.
ADMIN_ASSIGNABLE_PLANS = {"free", "friends", "pro"}


@router.post("/{user_id}/plan", response_model=CustomerDetail)
@limiter.limit("20/minute")
def update_customer_plan(
    request: Request,
    user_id: int,
    data: UpdateCustomerPlanRequest,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin),
):
    if data.new_plan not in ADMIN_ASSIGNABLE_PLANS:
        raise HTTPException(status_code=400, detail="Invalid plan")

    customer = db.query(User).filter(User.id == user_id).first()
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")

    old_plan = customer.plan
    updated_user = update_user_plan(db=db, user_id=user_id, new_plan=data.new_plan)

    create_note(
        db,
        user_id=user_id,
        admin_author_id=current_admin.id,
        note_text=f"Plan geändert von '{old_plan}' zu '{data.new_plan}' durch Admin ({current_admin.email}).",
    )
    return updated_user


@router.post("/{user_id}/reset-usage", response_model=CustomerDetail)
@limiter.limit("20/minute")
def reset_customer_usage(
    request: Request,
    user_id: int,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin),
):
    customer = db.query(User).filter(User.id == user_id).first()
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")

    updated_user = reset_monthly_request_count(db, user_id)

    create_note(
        db,
        user_id=user_id,
        admin_author_id=current_admin.id,
        note_text=f"Verbrauchszähler manuell zurückgesetzt durch Admin ({current_admin.email}).",
    )
    return updated_user


@router.delete("/{user_id}", status_code=204)
@limiter.limit("20/minute")
def delete_customer(
    request: Request,
    user_id: int,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_admin),
):
    """Admin-Löschung eines Kunden-Kontos (Hard Delete, gleiche Logik wie der
    Self-Service-DSGVO-Flow). Bewusst KEINE create_note: die Notizen des
    Kunden werden im selben Flow mitgelöscht — der Audit-Trail ist das
    'account_deleted'-Event (überlebt via ON DELETE SET NULL) mit
    Admin-Kennung im Metadata."""
    customer = db.query(User).filter(User.id == user_id).first()
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")

    if customer.id == current_admin.id:
        raise HTTPException(status_code=400, detail="You cannot delete your own account")

    if customer.plan == "admin":
        # Defense-in-Depth analog zu ADMIN_ASSIGNABLE_PLANS: Admin-Konten
        # werden ausschliesslich out-of-band (scripts/set_admin.py) verwaltet.
        raise HTTPException(status_code=400, detail="Admin accounts cannot be deleted via the CRM")

    try:
        delete_user_account(
            db,
            customer,
            event_metadata={
                "deleted_by": "admin",
                "admin_id": current_admin.id,
                "admin_email": current_admin.email,
            },
        )
    except StripeCancellationError:
        raise HTTPException(
            status_code=502,
            detail="Stripe cancellation failed; account was not deleted",
        )
