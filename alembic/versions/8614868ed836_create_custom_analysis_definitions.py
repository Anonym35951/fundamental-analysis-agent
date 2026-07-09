"""create custom_analysis_definitions table

Revision ID: 8614868ed836
Revises: 67b6fc384dc2
Create Date: 2026-06-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8614868ed836'
down_revision: Union[str, Sequence[str], None] = '67b6fc384dc2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'custom_analysis_definitions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('metrics', sa.JSON(), nullable=False),
        sa.Column('last_run_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_custom_analysis_definitions_id'),
        'custom_analysis_definitions', ['id'], unique=False,
    )
    op.create_index(
        op.f('ix_custom_analysis_definitions_user_id'),
        'custom_analysis_definitions', ['user_id'], unique=False,
    )

    op.add_column(
        'analysis_history',
        sa.Column('definition_id', sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        'fk_analysis_history_definition_id',
        'analysis_history', 'custom_analysis_definitions',
        ['definition_id'], ['id'],
        ondelete='SET NULL',
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('fk_analysis_history_definition_id', 'analysis_history', type_='foreignkey')
    op.drop_column('analysis_history', 'definition_id')

    op.drop_index(op.f('ix_custom_analysis_definitions_user_id'), table_name='custom_analysis_definitions')
    op.drop_index(op.f('ix_custom_analysis_definitions_id'), table_name='custom_analysis_definitions')
    op.drop_table('custom_analysis_definitions')
