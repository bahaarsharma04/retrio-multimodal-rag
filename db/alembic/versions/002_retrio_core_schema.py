"""Add projects, documents, pipeline_runs, pipeline_stages, document_chunks, message_citations tables

Revision ID: 002_retrio_core_schema
Revises: 001_initial_schema
Create Date: 2026-11-16 15:20:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    from sqlalchemy.types import UserDefinedType
    class Vector(UserDefinedType):
        def __init__(self, dim):
            self.dim = dim
        def get_col_spec(self, **kw):
            return f"vector({self.dim})"

# revision identifiers, used by Alembic.
revision = '002_retrio_core_schema'
down_revision = '001_initial_schema'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 0. Enable pgvector if dialect is PostgreSQL
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
        op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto";')
        try:
            op.execute('CREATE EXTENSION IF NOT EXISTS "vector";')
        except Exception:
            pass

    # 1. Projects table
    op.create_table(
        'projects',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), server_default='', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_projects_created_at'), 'projects', ['created_at'], unique=False)

    # 2. Documents table
    op.create_table(
        'documents',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('project_id', sa.String(length=64), nullable=False),
        sa.Column('filename', sa.String(length=512), nullable=False),
        sa.Column('file_path', sa.String(length=1024), nullable=False),
        sa.Column('file_hash', sa.String(length=64), nullable=False),
        sa.Column('mime_type', sa.String(length=128), server_default='application/pdf', nullable=False),
        sa.Column('ingestion_status', sa.String(length=50), server_default='pending', nullable=False),
        sa.Column('total_pages', sa.Integer(), server_default='1', nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('project_id', 'filename', name='uq_project_document_name')
    )
    op.create_index(op.f('ix_documents_project_id'), 'documents', ['project_id'], unique=False)
    op.create_index(op.f('ix_documents_file_hash'), 'documents', ['file_hash'], unique=False)
    op.create_index(op.f('ix_documents_ingestion_status'), 'documents', ['ingestion_status'], unique=False)

    # 3. Pipeline Runs table
    op.create_table(
        'pipeline_runs',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('document_id', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=50), server_default='started', nullable=False),
        sa.Column('total_chunks', sa.Integer(), server_default='0', nullable=False),
        sa.Column('duration_seconds', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_pipeline_runs_document_id'), 'pipeline_runs', ['document_id'], unique=False)

    # 4. Pipeline Stages table
    op.create_table(
        'pipeline_stages',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('pipeline_run_id', sa.String(length=64), nullable=False),
        sa.Column('stage_name', sa.String(length=100), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('duration_ms', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('stats', sa.JSON(), nullable=True),
        sa.Column('checkpoint_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['pipeline_run_id'], ['pipeline_runs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_pipeline_stages_pipeline_run_id'), 'pipeline_stages', ['pipeline_run_id'], unique=False)

    # 5. Document Chunks table (with pgvector embedding)
    op.create_table(
        'document_chunks',
        sa.Column('id', sa.String(length=128), nullable=False),
        sa.Column('document_id', sa.String(length=64), nullable=False),
        sa.Column('project_id', sa.String(length=64), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('chunk_type', sa.String(length=50), server_default='text', nullable=False),
        sa.Column('section', sa.String(length=255), server_default='General', nullable=False),
        sa.Column('page_numbers', sa.JSON(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('text_as_html', sa.Text(), nullable=True),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('indexed_text', sa.Text(), nullable=True),
        sa.Column('content_hash', sa.String(length=64), nullable=False),
        sa.Column('char_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_document_chunks_document_id'), 'document_chunks', ['document_id'], unique=False)
    op.create_index(op.f('ix_document_chunks_project_id'), 'document_chunks', ['project_id'], unique=False)
    op.create_index(op.f('ix_document_chunks_chunk_type'), 'document_chunks', ['chunk_type'], unique=False)
    op.create_index(op.f('ix_document_chunks_section'), 'document_chunks', ['section'], unique=False)

    # 6. Alter Conversations: add active_filters column
    try:
        op.add_column('conversations', sa.Column('active_filters', sa.JSON(), nullable=True))
    except Exception:
        pass

    # 7. Alter Messages: add rewritten_query and latency_ms
    try:
        op.add_column('messages', sa.Column('rewritten_query', sa.Text(), nullable=True))
        op.add_column('messages', sa.Column('latency_ms', sa.Float(), server_default='0.0', nullable=False))
    except Exception:
        pass

    # 8. Message Citations table
    op.create_table(
        'message_citations',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('message_id', sa.BigInteger(), nullable=False),
        sa.Column('chunk_id', sa.String(length=128), nullable=True),
        sa.Column('document_id', sa.String(length=64), nullable=True),
        sa.Column('citation_index', sa.Integer(), nullable=False),
        sa.Column('page_number', sa.Integer(), server_default='1', nullable=False),
        sa.Column('section', sa.String(length=255), nullable=True),
        sa.Column('relevance_score', sa.Float(), nullable=True),
        sa.Column('retrieval_method', sa.String(length=50), nullable=True),
        sa.Column('snippet_preview', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['message_id'], ['messages.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['chunk_id'], ['document_chunks.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_message_citations_message_id'), 'message_citations', ['message_id'], unique=False)


def downgrade() -> None:
    op.drop_table('message_citations')
    op.drop_column('messages', 'latency_ms')
    op.drop_column('messages', 'rewritten_query')
    op.drop_column('conversations', 'active_filters')
    op.drop_table('document_chunks')
    op.drop_table('pipeline_stages')
    op.drop_table('pipeline_runs')
    op.drop_table('documents')
    op.drop_table('projects')
