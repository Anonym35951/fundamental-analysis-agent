from datetime import datetime
from typing import Literal

from pydantic import BaseModel, computed_field


class MetricCriterion(BaseModel):
    operator: Literal[">", "<", ">=", "<="]
    threshold: float


class MetricSelection(BaseModel):
    key: str
    params: dict = {}
    criterion: MetricCriterion | None = None


class CustomAnalysisDefinitionCreate(BaseModel):
    name: str
    metrics: list[MetricSelection]


class CustomAnalysisDefinitionUpdate(BaseModel):
    name: str | None = None
    metrics: list[MetricSelection] | None = None


class CustomAnalysisDefinitionRead(BaseModel):
    id: int
    name: str
    metrics: list[MetricSelection]
    last_run_at: datetime | None
    created_at: datetime
    updated_at: datetime

    @computed_field
    @property
    def metric_count(self) -> int:
        return len(self.metrics)

    class Config:
        from_attributes = True


class CustomAnalysisDefinitionRunRequest(BaseModel):
    symbol: str
    frequency: str | None = None
