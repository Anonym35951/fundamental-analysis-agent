from datetime import datetime

from pydantic import BaseModel


class FavoriteEntry(BaseModel):
    symbol: str
    created_at: datetime

    class Config:
        from_attributes = True
