"""add profile and consent fields to users

Revision ID: 108252fde6af
Revises: 917997e6de48
Create Date: 2026-07-03 06:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '108252fde6af'
down_revision: Union[str, Sequence[str], None] = '917997e6de48'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('users', sa.Column('username', sa.String(length=50), nullable=True))
    op.add_column('users', sa.Column('first_name', sa.String(length=100), nullable=True))
    op.add_column('users', sa.Column('last_name', sa.String(length=100), nullable=True))
    op.add_column('users', sa.Column('age', sa.Integer(), nullable=True))
    op.add_column('users', sa.Column('terms_accepted_at', sa.DateTime(), nullable=True))
    op.add_column('users', sa.Column('terms_version', sa.String(length=20), nullable=True))
    op.add_column('users', sa.Column('privacy_accepted_at', sa.DateTime(), nullable=True))
    op.add_column('users', sa.Column('privacy_version', sa.String(length=20), nullable=True))
    op.add_column('users', sa.Column('onboarding_completed_at', sa.DateTime(), nullable=True))
    op.create_index('ix_users_username', 'users', ['username'], unique=True)

    # Bestandsnutzer sollen die neue mehrseitige Onboarding-Tour nicht
    # nachtraeglich aufgezwungen bekommen -> als "bereits abgeschlossen"
    # markieren (created_at als plausibler historischer Zeitstempel).
    op.execute(
        "UPDATE users SET onboarding_completed_at = created_at "
        "WHERE onboarding_completed_at IS NULL"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_users_username', table_name='users')
    op.drop_column('users', 'onboarding_completed_at')
    op.drop_column('users', 'privacy_version')
    op.drop_column('users', 'privacy_accepted_at')
    op.drop_column('users', 'terms_version')
    op.drop_column('users', 'terms_accepted_at')
    op.drop_column('users', 'age')
    op.drop_column('users', 'last_name')
    op.drop_column('users', 'first_name')
    op.drop_column('users', 'username')
