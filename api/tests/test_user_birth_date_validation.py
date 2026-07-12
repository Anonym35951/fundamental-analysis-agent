"""Regressionstests für die Umstellung von `age` auf `birth_date` bei neuen
Registrierungen (LAUNCH.md-Feedback 2026-07-12): Geburtsdatum ersetzt eine
manuell eingegebene, jährlich veraltende Zahl. Bestandsnutzer mit gesetztem
Legacy-`age` sind bewusst nicht Teil dieser Tests - deren Daten bleiben
unverändert (siehe api/models/user.py-Kommentare).
"""
from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from api.schemas.user import MIN_AGE, UserCreate, UserProfileUpdateRequest, calculate_age


def _years_ago(years: int, *, extra_days: int = 0) -> date:
    """Kalendarisch korrektes `years` Jahre vor heute (nicht 365*years, das
    akkumulierte Schaltjahre über >1 Jahr Distanz vernachlaessigt und nahe
    der Mindestalter-Grenze off-by-one werden kann)."""
    today = date.today()
    try:
        result = today.replace(year=today.year - years)
    except ValueError:
        # 29. Februar in einem Nicht-Schaltjahr-Ziel.
        result = today.replace(month=2, day=28, year=today.year - years)
    return result - timedelta(days=extra_days)


def test_calculate_age_before_birthday_this_year():
    # Geburtstag liegt noch in der Zukunft (im laufenden Jahr) -> ein Jahr weniger.
    today = date(2026, 6, 1)
    birth = date(2000, 12, 31)
    assert calculate_age(birth, today=today) == 25


def test_calculate_age_after_birthday_this_year():
    today = date(2026, 6, 1)
    birth = date(2000, 1, 1)
    assert calculate_age(birth, today=today) == 26


def test_calculate_age_on_birthday_itself():
    today = date(2026, 6, 1)
    birth = date(2000, 6, 1)
    assert calculate_age(birth, today=today) == 26


def _valid_register_payload(birth_date: date) -> dict:
    return {
        "email": "test@example.com",
        "password": "supersecret123",
        "username": "test_user",
        "first_name": "Test",
        "last_name": "User",
        "birth_date": birth_date,
        "terms_accepted": True,
        "privacy_accepted": True,
    }


def test_register_accepts_birth_date_exactly_min_age():
    payload = _valid_register_payload(_years_ago(MIN_AGE, extra_days=1))
    created = UserCreate(**payload)
    assert created.birth_date == payload["birth_date"]


def test_register_rejects_birth_date_under_min_age():
    payload = _valid_register_payload(_years_ago(MIN_AGE - 1))
    with pytest.raises(ValidationError, match=f"Mindestalter ist {MIN_AGE} Jahre"):
        UserCreate(**payload)


def test_register_rejects_future_birth_date():
    payload = _valid_register_payload(date.today() + timedelta(days=1))
    with pytest.raises(ValidationError, match="darf nicht in der Zukunft liegen"):
        UserCreate(**payload)


def test_profile_update_birth_date_optional_and_validated():
    # Nicht gesetzt -> kein Fehler (Feld bleibt unberührt).
    UserProfileUpdateRequest()

    # Gesetzt, aber zu jung -> Fehler.
    with pytest.raises(ValidationError, match=f"Mindestalter ist {MIN_AGE} Jahre"):
        UserProfileUpdateRequest(birth_date=_years_ago(MIN_AGE - 1))

    # Gesetzt und alt genug -> ok.
    valid = _years_ago(MIN_AGE + 5)
    updated = UserProfileUpdateRequest(birth_date=valid)
    assert updated.birth_date == valid
