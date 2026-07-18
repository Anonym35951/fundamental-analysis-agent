"""add locale to users

Revision ID: dc0e6ca04bbf
Revises: d4f6a8b0c2e4
Create Date: 2026-07-18 18:51:44.389889

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dc0e6ca04bbf'
down_revision: Union[str, Sequence[str], None] = 'd4f6a8b0c2e4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # EVOLVING.md § Internationalisierung, I18N-004: NULL = keine Präferenz
    # gesetzt, kein Default, kein Backfill. Bestandsnutzer verhalten sich
    # dadurch exakt wie vor dieser Migration (Sprache kommt weiterhin aus
    # localStorage/Browser-Erkennung, siehe frontend/src/i18n/detect.ts).
    op.add_column('users', sa.Column('locale', sa.String(length=10), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('users', 'locale')
