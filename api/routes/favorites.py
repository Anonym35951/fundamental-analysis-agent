from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.core.dependencies import get_current_user, get_db
from api.crud.favorite import add_favorite, get_favorites, remove_favorite
from api.models.user import User
from api.schemas.favorite import FavoriteEntry

router = APIRouter(prefix="/favorites", tags=["favorites"])


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
    return add_favorite(db, current_user.id, symbol)


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
