"""Test-Setup für die API-Schicht (getrennt von agent/tests/, das die
Analyse-Engine gegen eingefrorene SEC-Fixtures prüft).

api/core/config.py verlangt beim Import mehrere Pflicht-Env-Vars (siehe
LAUNCH.md P0-3 - genau das ließ den Render-Deploy ohne gesetzte Env-Vars
abstürzen). Damit diese Tests unabhängig von einer lokalen .env
deterministisch laufen (z. B. in CI, wo keine .env existiert), werden hier
Dummy-Werte gesetzt, BEVOR irgendein api.*-Modul importiert wird - das muss
auf Modulebene passieren (nicht in einer Fixture-Funktion), da pytest
conftest.py vor der Testsammlung importiert, test_*.py-Module aber schon
beim Import `from api...` ausführen.
"""
import os

_DUMMY_ENV = {
    "DATABASE_URL": "sqlite:///:memory:",
    "SECRET_KEY": "test-secret-key-not-used-for-real-auth",
    "FRONTEND_URL": "http://testserver",
    "STRIPE_SECRET_KEY": "sk_test_dummy",
    "STRIPE_WEBHOOK_SECRET": "whsec_dummy",
    "STRIPE_PRICE_ID_PRO_MONTHLY": "price_dummy_monthly",
    "STRIPE_PRICE_ID_PRO_YEARLY": "price_dummy_yearly",
    "STRIPE_SUCCESS_URL": "http://testserver/success",
    "STRIPE_CANCEL_URL": "http://testserver/cancel",
    "EMAIL_FROM": "test@example.com",
    "RESEND_API_KEY": "re_test_dummy",
}
for _key, _value in _DUMMY_ENV.items():
    os.environ[_key] = _value

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.models.base import Base
# Modelle importieren, damit Base.metadata ihre Tabellen kennt. Keine
# ORM-relationship()-Verknüpfungen zwischen User/ProductEvent (nur eine
# rohe FK-Spalte) - andere Modelle müssen hier nicht registriert werden.
from api.models.user import User
from api.models.product_event import ProductEvent


@pytest.fixture
def db():
    """Frische In-Memory-SQLite-DB pro Test - kein Netzwerk, keine echte
    Postgres-Instanz nötig, keine Testdaten überleben zwischen Tests.

    create_all() bekommt bewusst eine explizite tables=[...]-Liste statt
    "alles in Base.metadata": Base.metadata ist ein prozessweit geteiltes
    Singleton, das sich mit jedem `from api.models.X import Y` in JEDER
    Testdatei füllt (auch in anderen als dieser) - importiert z. B. ein Test
    `api/routes/custom_analysis.py`, registriert das transitiv auch
    `AnalysisHistory` (Postgres-`JSONB`-Spalte), was create_all() auf SQLite
    zum Absturz bringt. Die explizite Liste macht diese Fixture unabhängig
    davon, was anderswo im selben Testlauf sonst noch importiert wurde.
    """
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine, tables=[User.__table__, ProductEvent.__table__])
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_local()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
