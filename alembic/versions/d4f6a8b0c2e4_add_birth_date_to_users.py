"""add birth_date to users

Revision ID: d4f6a8b0c2e4
Revises: 9a4f8ab8c4c6
Create Date: 2026-07-12 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4f6a8b0c2e4'
down_revision: Union[str, Sequence[str], None] = '9a4f8ab8c4c6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Ersetzt `age` für neue Registrierungen (siehe api/models/user.py) -
    # bestehende `age`-Werte bleiben unverändert, kein Backfill.
    op.add_column('users', sa.Column('birth_date', sa.Date(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('users', 'birth_date')
