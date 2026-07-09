"""create customer_notes table

Revision ID: 0de6753e5c58
Revises: 108252fde6af
Create Date: 2026-07-03 07:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0de6753e5c58'
down_revision: Union[str, Sequence[str], None] = '108252fde6af'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'customer_notes',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('admin_author_id', sa.Integer(), nullable=True),
        sa.Column('note', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['admin_author_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_customer_notes_id', 'customer_notes', ['id'])
    op.create_index(
        'ix_customer_notes_user_id_created_at',
        'customer_notes',
        ['user_id', 'created_at'],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_customer_notes_user_id_created_at', table_name='customer_notes')
    op.drop_index('ix_customer_notes_id', table_name='customer_notes')
    op.drop_table('customer_notes')
