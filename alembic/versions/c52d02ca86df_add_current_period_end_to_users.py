"""add current_period_end to users

Revision ID: c52d02ca86df
Revises: 528fbc73a063
Create Date: 2026-04-19 10:59:19.185859

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'c52d02ca86df'
down_revision: Union[str, Sequence[str], None] = '528fbc73a063'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('users', sa.Column('current_period_end', sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column('users', 'current_period_end')
