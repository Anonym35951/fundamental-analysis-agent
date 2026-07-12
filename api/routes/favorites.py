from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.core.dependencies import get_current_user, get_db
from api.crud.favorite import add_favorite, get_favorites, remove_favorite
from api.models.favorite import Favorite
from api.models.symbol import Symbol
from api.models.user import User
from api.schemas.favorite import FavoriteEntry
from api.services.event_service import log_event

router = APIRouter(prefix="/favorites", tags=["favorites"])

# Obergrenze pro Nutzer (LAUNCH_AUDIT.md P2-8) - Ziel ist Abuse-Schutz für
# den periodischen SEC-Filing-Worker (filing_alert_service.py iteriert alle
# distinct Favoriten-Symbole mit einem festen Delay pro Symbol), nicht eine
# Produkt-Limitierung. 50 ist grosszügig über jedem realistischen
# Nutzungsmuster.
FAVORITES_LIMIT_PER_USER = 50


@router.get("", response_model=list[FavoriteEntry])
def list_favorites(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_favorites(db, current_user.id)


@router.post("", response_model=FavoriteEntry, status_code=status.HTTP_201_CREATED)
def create_favorite(
    symbol: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not symbol or not symbol.strip():
        raise HTTPException(status_code=400, detail="Symbol darf nicht leer sein.")

    normalized = symbol.strip().upper()

    # Limit/Validierung nur für tatsächlich NEUE Favoriten prüfen -
    # add_favorite ist idempotent, ein Re-Add eines bereits vorhandenen
    # Favoriten (z. B. Doppelklick) darf nicht am Limit scheitern.
    already_favorited = (
        db.query(Favorite)
        .filter(Favorite.user_id == current_user.id, Favorite.symbol == normalized)
        .first()
        is not None
    )
    if not already_favorited:
        current_count = db.query(Favorite).filter(Favorite.user_id == current_user.id).count()
        if current_count >= FAVORITES_LIMIT_PER_USER:
            raise HTTPException(
                status_code=400,
                detail=f"Maximal {FAVORITES_LIMIT_PER_USER} Favoriten pro Konto.",
            )

        symbol_exists = db.query(Symbol.id).filter(Symbol.symbol.ilike(normalized)).first() is not None
        if not symbol_exists:
            raise HTTPException(status_code=400, detail="Unbekanntes Symbol.")

    favorite = add_favorite(db, current_user.id, symbol)
    log_event(db, "favorite_added", user_id=current_user.id, metadata={"symbol": favorite.symbol})
    return favorite


@router.delete("/{symbol}", status_code=status.HTTP_204_NO_CONTENT)
def delete_favorite(
    symbol: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    removed = remove_favorite(db, current_user.id, symbol)
    if not removed:
        raise HTTPException(status_code=404, detail="Favorit nicht gefunden.")
    return None
