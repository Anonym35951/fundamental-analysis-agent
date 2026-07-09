"""add user plan and limits

Revision ID: c27e9f93a90e
Revises: 5e2265dceee9
Create Date: 2026-04-06 15:56:27.896827

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c27e9f93a90e'
down_revision: Union[str, Sequence[str], None] = '5e2265dceee9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'users',
        sa.Column('plan', sa.String(length=50), nullable=False, server_default='free')
    )
    op.add_column(
        'users',
        sa.Column('monthly_request_count', sa.Integer(), nullable=False, server_default='0')
    )
    op.add_column(
        'users',
        sa.Column('monthly_request_limit', sa.Integer(), nullable=False, server_default='100')
    )
    op.add_column(
        'users',
        sa.Column('stripe_customer_id', sa.String(length=255), nullable=True)
    )
    op.add_column(
        'users',
        sa.Column('stripe_subscription_id', sa.String(length=255), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('users', 'stripe_subscription_id')
    op.drop_column('users', 'stripe_customer_id')
    op.drop_column('users', 'monthly_request_limit')
    op.drop_column('users', 'monthly_request_count')
    op.drop_column('users', 'plan')