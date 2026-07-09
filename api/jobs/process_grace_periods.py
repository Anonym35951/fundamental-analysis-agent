import logging
from datetime import datetime, UTC

from api.core.database import SessionLocal
from api.models.user import User
from api.services.user_service import update_user_plan

logger = logging.getLogger(__name__)


def process_grace_periods():
    db = SessionLocal()

    try:
        now = datetime.now(UTC)

        users = (
            db.query(User)
            .filter(
                User.billing_status == "past_due",
                User.grace_until != None,
                User.grace_until < now,
            )
            .all()
        )

        logger.info("Found %s users to downgrade", len(users))

        for user in users:
            updated_user = update_user_plan(
                db=db,
                user_id=user.id,
                new_plan="free",
                stripe_customer_id=user.stripe_customer_id,
                stripe_subscription_id=None,
            )

            if updated_user is not None:
                updated_user.billing_status = "canceled"
                updated_user.grace_until = None

                db.commit()
                db.refresh(updated_user)

                logger.info(
                    "User downgraded after grace period: user_id=%s", user.id
                )

    finally:
        db.close()


if __name__ == "__main__":
    process_grace_periods()