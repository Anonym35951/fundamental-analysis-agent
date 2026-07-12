"""Regressionstest für LAUNCH_AUDIT.md P2-11: Vollanalyse kostete pauschal 1
Quota-Einheit, obwohl sie intern len(plan) (~10) Teil-Analysen berechnet -
dieselbe Rechenlast wie eine Custom-Analyse mit 10 Kennzahlen (die 10
Einheiten gekostet hätte). Betreiber-Entscheidung 2026-07-12: Vollanalyse an
die tatsächliche Last angleichen.

jobs.submit wird gemockt, damit der Job nicht wirklich im Thread-Pool läuft
(kein Netzwerkzugriff/keine echte Analyse in diesem Test - wir prüfen nur
den Quota-Verbrauch beim Start).
"""
from unittest.mock import MagicMock

import pytest
from starlette.requests import Request

import api.routes.full_analysis as full_analysis_module
from api.models.analysis_history import AnalysisHistory
from api.models.base import Base
from api.models.user import User
from api.routes.full_analysis import build_analysis_plan, start_full_analysis


@pytest.fixture
def db_full(db):
    Base.metadata.create_all(db.get_bind(), tables=[AnalysisHistory.__table__])
    return db


def _make_user(db, **kwargs) -> User:
    defaults = {
        "email": f"user{id(kwargs)}@example.com",
        "hashed_password": "x",
        "plan": "free",
        "email_verified": True,
        "monthly_request_limit": 50,
        "monthly_request_count": 0,
    }
    defaults.update(kwargs)
    user = User(**defaults)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _fake_request() -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/full/start",
        "headers": [],
        "client": ("testclient", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
        "query_string": b"",
    }
    return Request(scope)


def test_full_analysis_consumes_units_matching_plan_size(db_full, monkeypatch):
    monkeypatch.setattr(full_analysis_module.jobs, "submit", MagicMock())
    user = _make_user(db_full)
    expected_units = len(build_analysis_plan())
    assert expected_units > 1  # Regressions-Anker: der Bug war genau, dass dies ignoriert wurde.

    start_full_analysis(request=_fake_request(), symbol="AAPL", current_user=user, db=db_full)

    db_full.refresh(user)
    assert user.monthly_request_count == expected_units
