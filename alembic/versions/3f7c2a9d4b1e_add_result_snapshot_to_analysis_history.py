"""add result_snapshot to analysis_history

Revision ID: 3f7c2a9d4b1e
Revises: 5a2efe2e66f4
Create Date: 2026-06-25 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '3f7c2a9d4b1e'
down_revision: Union[str, Sequence[str], None] = '5a2efe2e66f4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'analysis_history',
        sa.Column('result_snapshot', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('analysis_history', 'result_snapshot')
