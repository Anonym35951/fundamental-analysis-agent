"""add billing interval and stripe price id to users

Revision ID: 528fbc73a063
Revises: c992ae92f4d4
Create Date: 2026-04-17 18:04:09.353551

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '528fbc73a063'
down_revision: Union[str, Sequence[str], None] = 'c992ae92f4d4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('users', sa.Column('billing_interval', sa.String(length=20), nullable=True))
    op.add_column('users', sa.Column('stripe_price_id', sa.String(length=255), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('users', 'stripe_price_id')
    op.drop_column('users', 'billing_interval')