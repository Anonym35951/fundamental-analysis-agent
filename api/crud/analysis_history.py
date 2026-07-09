from sqlalchemy.orm import Session

from api.models.analysis_history import AnalysisHistory


def create_history_entry(
    db: Session,
    user_id: int,
    job_id: str,
    symbol: str,
    mode: str,
    frequency: str | None,
) -> AnalysisHistory:
    entry = AnalysisHistory(
        user_id=user_id,
        job_id=job_id,
        symbol=symbol,
        mode=mode,
        frequency=frequency,
        status="running",
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def update_history_status(
    db: Session,
    job_id: str,
    status: str,
    snapshot: dict | None = None,
) -> None:
    values: dict = {AnalysisHistory.status: status}
    if snapshot is not None:
        values[AnalysisHistory.result_snapshot] = snapshot
    db.query(AnalysisHistory).filter(AnalysisHistory.job_id == job_id).update(values)
    db.commit()


def get_recent_history(db: Session, user_id: int, limit: int = 10) -> list[AnalysisHistory]:
    return (
        db.query(AnalysisHistory)
        .filter(AnalysisHistory.user_id == user_id)
        .order_by(AnalysisHistory.created_at.desc())
        .limit(limit)
        .all()
    )


def get_history_entry(db: Session, user_id: int, history_id: int) -> AnalysisHistory | None:
    return (
        db.query(AnalysisHistory)
        .filter(AnalysisHistory.id == history_id, AnalysisHistory.user_id == user_id)
        .first()
    )


def get_usage_summary(db: Session, user_id: int) -> dict:
    """Neutraler Nutzungs-Rückblick (Anzahl Analysen, Anzahl unterschiedlicher
    Unternehmen) — Grundlage für den Kündigungs-Flow (kein Guilt-Tripping,
    reine Information: siehe [[comanalysis-positioning]])."""
    total_analyses = (
        db.query(AnalysisHistory).filter(AnalysisHistory.user_id == user_id).count()
    )
    distinct_symbols = (
        db.query(AnalysisHistory.symbol)
        .filter(AnalysisHistory.user_id == user_id)
        .distinct()
        .count()
    )
    return {"total_analyses": total_analyses, "distinct_symbols": distinct_symbols}
