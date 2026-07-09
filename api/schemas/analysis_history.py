from datetime import datetime

from pydantic import BaseModel


class AnalysisHistoryEntry(BaseModel):
    id: int
    job_id: str
    symbol: str
    mode: str
    frequency: str | None
    status: str
    definition_id: int | None
    created_at: datetime

    class Config:
        from_attributes = True


class AnalysisHistorySnapshot(AnalysisHistoryEntry):
    result_snapshot: dict | None
