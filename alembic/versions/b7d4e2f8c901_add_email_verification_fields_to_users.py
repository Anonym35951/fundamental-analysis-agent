"""add email verification fields to users

Revision ID: b7d4e2f8c901
Revises: a1b2c3d4e5f6
Create Date: 2026-07-02 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'b7d4e2f8c901'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Bestandskonten (nur Testkonten) bewusst NICHT grandfathern:
    # alle müssen verifizieren.
    op.add_column(
        'users',
        sa.Column('email_verified', sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column('users', sa.Column('email_verification_token_hash', sa.String(255), nullable=True))
    op.add_column('users', sa.Column('email_verification_expires', sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column('users', 'email_verification_expires')
    op.drop_column('users', 'email_verification_token_hash')
    op.drop_column('users', 'email_verified')
