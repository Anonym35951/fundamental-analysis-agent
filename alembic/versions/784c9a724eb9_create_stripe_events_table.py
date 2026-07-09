"""create stripe events table

Revision ID: 784c9a724eb9
Revises: f473c7505e08
Create Date: 2026-04-06

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "784c9a724eb9"
down_revision: Union[str, Sequence[str], None] = "f473c7505e08"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "stripe_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("event_id", sa.String(length=255), nullable=False),
        sa.Column("event_type", sa.String(length=255), nullable=False),
        sa.Column("processed_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_stripe_events_id"), "stripe_events", ["id"], unique=False)
    op.create_index(op.f("ix_stripe_events_event_id"), "stripe_events", ["event_id"], unique=True)


def downgrade() -> None:
    op.drop_index(op.f("ix_stripe_events_event_id"), table_name="stripe_events")
    op.drop_index(op.f("ix_stripe_events_id"), table_name="stripe_events")
    op.drop_table("stripe_events")