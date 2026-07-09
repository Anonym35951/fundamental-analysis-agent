import logging

from api.core.database import SessionLocal
from api.services.user_service import reset_all_free_users_monthly_request_counts

logger = logging.getLogger(__name__)


def run() -> None:
    db = SessionLocal()
    try:
        updated_rows = reset_all_free_users_monthly_request_counts(db)
        logger.info("Monthly reset completed. Reset free users: %s", updated_rows)
    finally:
        db.close()


if __name__ == "__main__":
    run()