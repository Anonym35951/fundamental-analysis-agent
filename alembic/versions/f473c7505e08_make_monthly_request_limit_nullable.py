"""make monthly_request_limit nullable

Revision ID: f473c7505e08
Revises: c27e9f93a90e
Create Date: 2026-04-06 19:16:33.214256

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f473c7505e08'
down_revision: Union[str, Sequence[str], None] = 'c27e9f93a90e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column(
        'users',
        'monthly_request_limit',
        existing_type=sa.Integer(),
        nullable=True
    )

    # Bestehende User: alles außer free → kein Limit
    op.execute(
        "UPDATE users SET monthly_request_limit = NULL WHERE plan != 'free';"
    )


def downgrade() -> None:
    """Downgrade schema."""
    # NULL wieder auf Default setzen (z. B. 10)
    op.execute(
        "UPDATE users SET monthly_request_limit = 10 WHERE monthly_request_limit IS NULL;"
    )

    op.alter_column(
        'users',
        'monthly_request_limit',
        existing_type=sa.Integer(),
        nullable=False
    )
