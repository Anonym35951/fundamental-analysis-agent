"""Regressionstests für LAUNCH_AUDIT.md P2-8: Favorites hatten kein Limit
pro Nutzer und keine Validierung gegen die symbols-Tabelle - beliebige
Strings landeten im periodischen SEC-Filing-Worker (filing_alert_service.py,
0.3s Delay pro distinct Symbol).

Stil wie test_product_events_p2_17.py: direkte Funktionsaufrufe, In-Memory-
SQLite mit Favorite + Symbol Tabellen.
"""
import pytest
from fastapi import HTTPException

from api.models.base import Base
from api.models.favorite import Favorite
from api.models.symbol import Symbol
from api.models.user import User
from api.routes.favorites import FAVORITES_LIMIT_PER_USER, create_favorite


@pytest.fixture
def db_full(db):
    Base.metadata.create_all(db.get_bind(), tables=[Favorite.__table__, Symbol.__table__])
    db.add_all(
        [
            Symbol(symbol="AAPL", name="Apple Inc.", exchange="NASDAQ"),
            Symbol(symbol="PYPL", name="PayPal Holdings", exchange="NASDAQ"),
        ]
    )
    db.commit()
    return db


def _make_user(db, **kwargs) -> User:
    defaults = {"email": f"user{id(kwargs)}@example.com", "hashed_password": "x"}
    defaults.update(kwargs)
    user = User(**defaults)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def test_create_favorite_rejects_unknown_symbol(db_full):
    user = _make_user(db_full)

    with pytest.raises(HTTPException) as exc:
        create_favorite(symbol="ZZZNOTREAL", current_user=user, db=db_full)

    assert exc.value.status_code == 400
    assert db_full.query(Favorite).filter(Favorite.user_id == user.id).count() == 0


def test_create_favorite_accepts_known_symbol_case_insensitive(db_full):
    user = _make_user(db_full)

    favorite = create_favorite(symbol="aapl", current_user=user, db=db_full)

    assert favorite.symbol == "AAPL"


def test_create_favorite_enforces_per_user_limit(db_full):
    user = _make_user(db_full)
    # Limit künstlich erreichen, ohne 50 echte Symbols anzulegen: die
    # Limit-Prüfung zählt nur vorhandene Favorite-Zeilen, unabhängig davon,
    # ob die Symbole real in der Symbol-Tabelle stehen.
    db_full.add_all(
        [Favorite(user_id=user.id, symbol=f"SYM{i}") for i in range(FAVORITES_LIMIT_PER_USER)]
    )
    db_full.commit()

    with pytest.raises(HTTPException) as exc:
        create_favorite(symbol="PYPL", current_user=user, db=db_full)

    assert exc.value.status_code == 400
    assert "Maximal" in exc.value.detail


def test_create_favorite_reAdd_of_existing_bypasses_limit(db_full):
    """Ein Re-Add eines bereits vorhandenen Favoriten (z.B. Doppelklick)
    darf nicht am Limit scheitern, auch wenn der Nutzer exakt am Limit ist."""
    user = _make_user(db_full)
    db_full.add_all(
        [Favorite(user_id=user.id, symbol=f"SYM{i}") for i in range(FAVORITES_LIMIT_PER_USER - 1)]
    )
    db_full.add(Favorite(user_id=user.id, symbol="PYPL"))
    db_full.commit()

    # Jetzt exakt am Limit (50 Zeilen), aber PYPL ist bereits einer davon.
    favorite = create_favorite(symbol="PYPL", current_user=user, db=db_full)

    assert favorite.symbol == "PYPL"
