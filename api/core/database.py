from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.core.config import settings


# Engine (Verbindung zur DB)
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.SQL_ECHO,  # lokal per SQL_ECHO=true in .env aktivierbar
    pool_pre_ping=True,  # tote Verbindungen (z. B. nach Render-Idle) aussortieren
)

# Session Factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


# Dependency für FastAPI (später wichtig)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()