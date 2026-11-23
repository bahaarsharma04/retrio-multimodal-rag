"""Add users table and owner_id / user_id foreign keys

Revision ID: 003_add_users_and_ownership
Revises: 002_retrio_core_schema
Create Date: 2026-11-18 16:55:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '003_add_users_and_ownership'
down_revision = '002_retrio_core_schema'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('email', sa.String(length=255), nullable=False, unique=True),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=True, server_default=''),
        sa.Column('role', sa.String(length=32), nullable=False, server_default='user'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_users_email', 'users', ['email'], unique=True)

    # 2. Add foreign keys to existing tables
    op.add_column('projects', sa.Column('owner_id', sa.String(length=64), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=True))
    op.create_index('ix_projects_owner_id', 'projects', ['owner_id'])

    op.add_column('documents', sa.Column('user_id', sa.String(length=64), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=True))
    op.create_index('ix_documents_user_id', 'documents', ['user_id'])

    op.add_column('conversations', sa.Column('user_id', sa.String(length=64), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=True))
    op.create_index('ix_conversations_user_id', 'conversations', ['user_id'])

    op.add_column('document_chunks', sa.Column('user_id', sa.String(length=64), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=True))
    op.create_index('ix_document_chunks_user_id', 'document_chunks', ['user_id'])


def downgrade() -> None:
    op.drop_index('ix_document_chunks_user_id', table_name='document_chunks')
    op.drop_column('document_chunks', 'user_id')

    op.drop_index('ix_conversations_user_id', table_name='conversations')
    op.drop_column('conversations', 'user_id')

    op.drop_index('ix_documents_user_id', table_name='documents')
    op.drop_column('documents', 'user_id')

    op.drop_index('ix_projects_owner_id', table_name='projects')
    op.drop_column('projects', 'owner_id')

    op.drop_index('ix_users_email', table_name='users')
    op.drop_table('users')
