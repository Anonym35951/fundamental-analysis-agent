"""add billing status and grace until to users

Revision ID: c992ae92f4d4
Revises: 784c9a724eb9
Create Date: 2026-04-09 19:15:48.320926

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c992ae92f4d4'
down_revision: Union[str, Sequence[str], None] = '784c9a724eb9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "users",
        sa.Column(
            "billing_status",
            sa.String(length=50),
            nullable=False,
            server_default="active",
        ),
    )
    op.add_column(
        "users",
        sa.Column(
            "grace_until",
            sa.DateTime(),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("users", "grace_until")
    op.drop_column("users", "billing_status")