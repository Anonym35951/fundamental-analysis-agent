"""EVOLVING.md § Internationalisierung, I18N-004: users.locale ist additiv
und NULL-safe. Bestandsnutzer (keine Migration ausgefuehrt fuer sie /
locale nie gesetzt) muessen unveraendert funktionieren; PATCH /auth/profile
validiert locale gegen ALLOWED_LOCALES statt beliebige Strings anzunehmen.

Stil wie test_register_cta_attribution_ev124.py: direkte Funktionsaufrufe
gegen die In-Memory-SQLite-`db`-Fixture, kein TestClient.
"""
from datetime import date

import pytest
from pydantic import ValidationError

from api.crud.user import create_user, update_user_profile
from api.schemas.user import UserProfileUpdateRequest


def _make_user(db):
    return create_user(
        db,
        email="locale-test@example.com",
        password="hunter2hunter2",
        username="localetestuser",
        first_name="Test",
        last_name="User",
        birth_date=date(1990, 1, 1),
        terms_version="v1",
        privacy_version="v1",
    )


def test_new_user_has_no_locale_preference_by_default(db):
    """Bestandsnutzer-Simulation: locale bleibt NULL, solange niemand sie
    explizit setzt - kein serverseitiger Default, kein Backfill."""
    user = _make_user(db)
    assert user.locale is None


def test_patch_profile_sets_a_valid_locale(db):
    user = _make_user(db)

    updated = update_user_profile(db, user, UserProfileUpdateRequest(locale="en"))

    assert updated.locale == "en"


def test_patch_profile_without_locale_leaves_existing_preference_untouched(db):
    user = _make_user(db)
    update_user_profile(db, user, UserProfileUpdateRequest(locale="en"))

    # Ein Profil-Update, das locale nicht mitsendet (z.B. nur username),
    # darf die zuvor gesetzte Sprachpraeferenz nicht loeschen.
    updated = update_user_profile(db, user, UserProfileUpdateRequest(username="renamed"))

    assert updated.locale == "en"
    assert updated.username == "renamed"


def test_invalid_locale_is_rejected_at_the_schema_level():
    """422 vor jeder DB-Operation - kein Muell in der Spalte."""
    with pytest.raises(ValidationError):
        UserProfileUpdateRequest(locale="fr")


def test_null_locale_is_accepted_by_the_schema_and_leaves_preference_unset():
    request = UserProfileUpdateRequest()
    assert request.locale is None
