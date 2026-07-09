from datetime import datetime

from sqlalchemy.orm import Session

from api.models.custom_analysis_definition import CustomAnalysisDefinition


def create_definition(db: Session, user_id: int, name: str, metrics: list[dict]) -> CustomAnalysisDefinition:
    definition = CustomAnalysisDefinition(user_id=user_id, name=name, metrics=metrics)
    db.add(definition)
    db.commit()
    db.refresh(definition)
    return definition


def list_definitions_for_user(db: Session, user_id: int) -> list[CustomAnalysisDefinition]:
    return (
        db.query(CustomAnalysisDefinition)
        .filter(CustomAnalysisDefinition.user_id == user_id)
        .order_by(CustomAnalysisDefinition.created_at.desc())
        .all()
    )


def get_definition(db: Session, user_id: int, definition_id: int) -> CustomAnalysisDefinition | None:
    return (
        db.query(CustomAnalysisDefinition)
        .filter(
            CustomAnalysisDefinition.id == definition_id,
            CustomAnalysisDefinition.user_id == user_id,
        )
        .first()
    )


def update_definition(
    db: Session,
    definition: CustomAnalysisDefinition,
    name: str | None = None,
    metrics: list[dict] | None = None,
) -> CustomAnalysisDefinition:
    if name is not None:
        definition.name = name
    if metrics is not None:
        definition.metrics = metrics
    db.commit()
    db.refresh(definition)
    return definition


def mark_definition_run(db: Session, definition: CustomAnalysisDefinition) -> None:
    definition.last_run_at = datetime.utcnow()
    db.commit()


def delete_definition(db: Session, definition: CustomAnalysisDefinition) -> None:
    db.delete(definition)
    db.commit()


def count_definitions_for_user(db: Session, user_id: int) -> int:
    return (
        db.query(CustomAnalysisDefinition)
        .filter(CustomAnalysisDefinition.user_id == user_id)
        .count()
    )
