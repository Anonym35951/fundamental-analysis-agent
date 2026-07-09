from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from api.crud.custom_analysis_definition import count_definitions_for_user
from api.models.user import User

FREE_PLAN_DEFINITION_LIMIT = 1


def check_can_save_definition(db: Session, user: User) -> None:
    """Mirrors the plan-based limit-checking convention in
    api/core/dependencies.py::require_analysis_access (monthly request
    limit), applied here to saved Custom Analysis definitions instead."""
    if user.plan == "free" and count_definitions_for_user(db, user.id) >= FREE_PLAN_DEFINITION_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Free plan allows max 1 saved custom analysis. Upgrade to Pro.",
        )
