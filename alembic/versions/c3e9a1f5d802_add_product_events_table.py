"""add product_events table

Revision ID: c3e9a1f5d802
Revises: b7d4e2f8c901
Create Date: 2026-07-02 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'c3e9a1f5d802'
down_revision: Union[str, Sequence[str], None] = 'b7d4e2f8c901'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'product_events',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('event_type', sa.String(length=64), nullable=False),
        sa.Column('event_metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_product_events_id'), 'product_events', ['id'], unique=False)
    op.create_index(op.f('ix_product_events_event_type'), 'product_events', ['event_type'], unique=False)
    op.create_index(op.f('ix_product_events_created_at'), 'product_events', ['created_at'], unique=False)
    op.create_index(
        'ix_product_events_type_created_at', 'product_events', ['event_type', 'created_at'], unique=False
    )
    op.create_index(
        'ix_product_events_user_id_created_at', 'product_events', ['user_id', 'created_at'], unique=False
    )


def downgrade() -> None:
    op.drop_index('ix_product_events_user_id_created_at', table_name='product_events')
    op.drop_index('ix_product_events_type_created_at', table_name='product_events')
    op.drop_index(op.f('ix_product_events_created_at'), table_name='product_events')
    op.drop_index(op.f('ix_product_events_event_type'), table_name='product_events')
    op.drop_index(op.f('ix_product_events_id'), table_name='product_events')
    op.drop_table('product_events')
