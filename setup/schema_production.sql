-- Production PostgreSQL Schema for Memory Pipeline
-- Uses dedicated 'memory_pipeline' schema instead of public
-- Compatible with PostgreSQL 13+ with clean, reliable setup

-- ================================
-- Schema and Extensions Setup
-- ================================

-- Create dedicated schema
DROP SCHEMA IF EXISTS memory_pipeline CASCADE;
CREATE SCHEMA memory_pipeline;

-- Set search path to use our schema by default
SET search_path TO memory_pipeline, public;

-- Enable required extensions in public schema (standard practice)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" SCHEMA public;

-- ================================
-- Custom Types
-- ================================

CREATE TYPE memory_pipeline.memory_type AS ENUM ('fact', 'preference', 'event', 'routine', 'emotion');
CREATE TYPE memory_pipeline.memory_category AS ENUM ('work', 'food', 'health', 'social', 'schedule', 'entertainment', 'emotional');
CREATE TYPE memory_pipeline.temporal_relevance AS ENUM ('permanent', 'yearly', 'monthly', 'weekly', 'daily', 'temporary');

-- ================================
-- Core Tables
-- ================================

-- Users table
CREATE TABLE memory_pipeline.users (
    user_id VARCHAR(255) PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    settings JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN DEFAULT true,
    total_memories INTEGER DEFAULT 0,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Memories table
CREATE TABLE memory_pipeline.memories (
    id BIGSERIAL PRIMARY KEY,
    memory_id UUID DEFAULT public.uuid_generate_v4() UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL REFERENCES memory_pipeline.users(user_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    memory_type memory_pipeline.memory_type NOT NULL,
    category memory_pipeline.memory_category NOT NULL,
    importance_score DECIMAL(3,2) CHECK (importance_score >= 0 AND importance_score <= 1),
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    temporal_relevance memory_pipeline.temporal_relevance DEFAULT 'permanent',

    -- Enhanced metadata using PostgreSQL JSONB
    metadata JSONB DEFAULT '{}'::jsonb,
    entities TEXT[] DEFAULT '{}',
    valid_contexts TEXT[] DEFAULT '{}',
    invalid_contexts TEXT[] DEFAULT '{}',

    -- Temporal fields
    extracted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    last_accessed TIMESTAMP WITH TIME ZONE,
    access_count INTEGER DEFAULT 0,

    -- Source tracking
    source_message_id VARCHAR(255),
    source_conversation_id BIGINT,

    -- Version control
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT true
);

-- Conversations table
CREATE TABLE memory_pipeline.conversations (
    id BIGSERIAL PRIMARY KEY,
    conversation_id UUID DEFAULT public.uuid_generate_v4() UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL REFERENCES memory_pipeline.users(user_id) ON DELETE CASCADE,
    message_id VARCHAR(255),
    message TEXT NOT NULL,
    memories_extracted INTEGER DEFAULT 0,
    processing_time_ms DECIMAL(10,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    context_type VARCHAR(100),
    session_id VARCHAR(255)
);

-- Memory connections
CREATE TABLE memory_pipeline.memory_connections (
    id BIGSERIAL PRIMARY KEY,
    source_memory_id UUID NOT NULL REFERENCES memory_pipeline.memories(memory_id) ON DELETE CASCADE,
    target_memory_id UUID NOT NULL REFERENCES memory_pipeline.memories(memory_id) ON DELETE CASCADE,
    connection_type VARCHAR(50) NOT NULL,
    strength DECIMAL(3,2) CHECK (strength >= 0 AND strength <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(source_memory_id, target_memory_id, connection_type)
);

-- Memory access log
CREATE TABLE memory_pipeline.memory_access_log (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL REFERENCES memory_pipeline.users(user_id) ON DELETE CASCADE,
    memory_id UUID NOT NULL REFERENCES memory_pipeline.memories(memory_id) ON DELETE CASCADE,
    query_text TEXT,
    context_type VARCHAR(100),
    relevance_score DECIMAL(5,4),
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    response_time_ms DECIMAL(10,2),
    session_id VARCHAR(255)
);

-- RAG metrics
CREATE TABLE memory_pipeline.rag_metrics (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL REFERENCES memory_pipeline.users(user_id) ON DELETE CASCADE,
    query_type VARCHAR(100),
    context_type VARCHAR(100),
    memories_retrieved INTEGER,
    avg_relevance DECIMAL(5,4),
    response_time_ms DECIMAL(10,2),
    token_budget_used INTEGER,
    deduplication_applied BOOLEAN DEFAULT false,
    validation_applied BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ChromaDB sync status
CREATE TABLE memory_pipeline.chroma_sync_status (
    memory_id UUID PRIMARY KEY REFERENCES memory_pipeline.memories(memory_id) ON DELETE CASCADE,
    chroma_id VARCHAR(255),
    collection_name VARCHAR(255),
    embedding_model VARCHAR(100),
    last_synced TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    sync_status VARCHAR(20) DEFAULT 'pending',
    error_message TEXT,
    retry_count INTEGER DEFAULT 0
);

-- ================================
-- Performance Indexes
-- ================================

-- Primary user-based queries
CREATE INDEX idx_memories_user_active
ON memory_pipeline.memories(user_id, is_active) WHERE is_active = true;

-- Context filtering
CREATE INDEX idx_memories_user_category_type
ON memory_pipeline.memories(user_id, category, memory_type) WHERE is_active = true;

-- Temporal queries
CREATE INDEX idx_memories_user_created
ON memory_pipeline.memories(user_id, extracted_at DESC) WHERE is_active = true;

-- Importance-based retrieval
CREATE INDEX idx_memories_user_importance
ON memory_pipeline.memories(user_id, importance_score DESC, extracted_at DESC) WHERE is_active = true;

-- Expiration management
CREATE INDEX idx_memories_expires_at
ON memory_pipeline.memories(expires_at) WHERE expires_at IS NOT NULL AND is_active = true;

-- Entity-based searches
CREATE INDEX idx_memories_entities
ON memory_pipeline.memories USING GIN(entities);

-- Context arrays
CREATE INDEX idx_memories_valid_contexts
ON memory_pipeline.memories USING GIN(valid_contexts);

-- JSONB metadata queries
CREATE INDEX idx_memories_metadata
ON memory_pipeline.memories USING GIN(metadata);

-- User activity
CREATE INDEX idx_users_last_activity
ON memory_pipeline.users(last_activity DESC) WHERE is_active = true;

-- Conversation tracking
CREATE INDEX idx_conversations_user_created
ON memory_pipeline.conversations(user_id, created_at DESC);

-- Memory access analytics
CREATE INDEX idx_access_log_user_date
ON memory_pipeline.memory_access_log(user_id, accessed_at DESC);

-- ChromaDB sync monitoring
CREATE INDEX idx_chroma_sync_status
ON memory_pipeline.chroma_sync_status(sync_status, last_synced);

-- ================================
-- Functions and Triggers
-- ================================

-- Update timestamp function
CREATE OR REPLACE FUNCTION memory_pipeline.update_updated_at_column()
RETURNS TRIGGER AS $func$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END
$func$ LANGUAGE plpgsql;

-- Apply to memories table
CREATE TRIGGER update_memories_updated_at
    BEFORE UPDATE ON memory_pipeline.memories
    FOR EACH ROW
    EXECUTE FUNCTION memory_pipeline.update_updated_at_column();

-- Apply to users table
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON memory_pipeline.users
    FOR EACH ROW
    EXECUTE FUNCTION memory_pipeline.update_updated_at_column();

-- ================================
-- Constraints and Validations
-- ================================

-- Ensure memory connections don't create self-references
ALTER TABLE memory_pipeline.memory_connections
ADD CONSTRAINT check_no_self_reference
CHECK (source_memory_id != target_memory_id);

-- Ensure conversation processing time is reasonable
ALTER TABLE memory_pipeline.conversations
ADD CONSTRAINT check_reasonable_processing_time
CHECK (processing_time_ms >= 0 AND processing_time_ms < 300000);

-- Ensure RAG metrics are reasonable
ALTER TABLE memory_pipeline.rag_metrics
ADD CONSTRAINT check_reasonable_response_time
CHECK (response_time_ms >= 0 AND response_time_ms < 60000);

-- Ensure sync retry count is reasonable
ALTER TABLE memory_pipeline.chroma_sync_status
ADD CONSTRAINT check_reasonable_retry_count
CHECK (retry_count >= 0 AND retry_count <= 10);

-- ================================
-- Default Permissions
-- ================================

-- Grant usage on schema to application user
-- GRANT USAGE ON SCHEMA memory_pipeline TO app_user;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA memory_pipeline TO app_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA memory_pipeline TO app_user;

-- Note: Uncomment and modify the above grants based on your specific user setup

-- ================================
-- Verification Queries
-- ================================

-- Verify schema creation
SELECT schemaname, tablename, tableowner
FROM pg_tables
WHERE schemaname = 'memory_pipeline'
ORDER BY tablename;

-- Verify indexes
SELECT schemaname, indexname, tablename
FROM pg_indexes
WHERE schemaname = 'memory_pipeline'
ORDER BY tablename, indexname;

-- Show schema usage
SHOW search_path;