-- Simple PostgreSQL Schema for Memory Pipeline
-- Compatible with PostgreSQL 13+ without advanced features

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(255) PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    settings JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN DEFAULT true,
    total_memories INTEGER DEFAULT 0,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Memory types and categories
CREATE TYPE memory_type AS ENUM ('fact', 'preference', 'event', 'routine', 'emotion');
CREATE TYPE memory_category AS ENUM ('work', 'food', 'health', 'social', 'schedule', 'entertainment', 'emotional');
CREATE TYPE temporal_relevance AS ENUM ('permanent', 'yearly', 'monthly', 'weekly', 'daily', 'temporary');

-- Memories table
CREATE TABLE IF NOT EXISTS memories (
    id BIGSERIAL PRIMARY KEY,
    memory_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    memory_type memory_type NOT NULL,
    category memory_category NOT NULL,
    importance_score DECIMAL(3,2) CHECK (importance_score >= 0 AND importance_score <= 1),
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    temporal_relevance temporal_relevance DEFAULT 'permanent',

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
CREATE TABLE IF NOT EXISTS conversations (
    id BIGSERIAL PRIMARY KEY,
    conversation_id UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    message_id VARCHAR(255),
    message TEXT NOT NULL,
    memories_extracted INTEGER DEFAULT 0,
    processing_time_ms DECIMAL(10,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    context_type VARCHAR(100),
    session_id VARCHAR(255)
);

-- Memory connections
CREATE TABLE IF NOT EXISTS memory_connections (
    id BIGSERIAL PRIMARY KEY,
    source_memory_id UUID NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
    target_memory_id UUID NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
    connection_type VARCHAR(50) NOT NULL,
    strength DECIMAL(3,2) CHECK (strength >= 0 AND strength <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(source_memory_id, target_memory_id, connection_type)
);

-- Memory access log
CREATE TABLE IF NOT EXISTS memory_access_log (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    memory_id UUID NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
    query_text TEXT,
    context_type VARCHAR(100),
    relevance_score DECIMAL(5,4),
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    response_time_ms DECIMAL(10,2),
    session_id VARCHAR(255)
);

-- RAG metrics
CREATE TABLE IF NOT EXISTS rag_metrics (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
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
CREATE TABLE IF NOT EXISTS chroma_sync_status (
    memory_id UUID PRIMARY KEY REFERENCES memories(memory_id) ON DELETE CASCADE,
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
CREATE INDEX IF NOT EXISTS idx_memories_user_active
ON memories(user_id, is_active) WHERE is_active = true;

-- Context filtering
CREATE INDEX IF NOT EXISTS idx_memories_user_category_type
ON memories(user_id, category, memory_type) WHERE is_active = true;

-- Temporal queries
CREATE INDEX IF NOT EXISTS idx_memories_user_created
ON memories(user_id, extracted_at DESC) WHERE is_active = true;

-- Importance-based retrieval
CREATE INDEX IF NOT EXISTS idx_memories_user_importance
ON memories(user_id, importance_score DESC, extracted_at DESC) WHERE is_active = true;

-- Expiration management
CREATE INDEX IF NOT EXISTS idx_memories_expires_at
ON memories(expires_at) WHERE expires_at IS NOT NULL AND is_active = true;

-- Entity-based searches
CREATE INDEX IF NOT EXISTS idx_memories_entities
ON memories USING GIN(entities);

-- Context arrays
CREATE INDEX IF NOT EXISTS idx_memories_valid_contexts
ON memories USING GIN(valid_contexts);

-- JSONB metadata queries
CREATE INDEX IF NOT EXISTS idx_memories_metadata
ON memories USING GIN(metadata);

-- User activity
CREATE INDEX IF NOT EXISTS idx_users_last_activity
ON users(last_activity DESC) WHERE is_active = true;

-- Conversation tracking
CREATE INDEX IF NOT EXISTS idx_conversations_user_created
ON conversations(user_id, created_at DESC);

-- Memory access analytics
CREATE INDEX IF NOT EXISTS idx_access_log_user_date
ON memory_access_log(user_id, accessed_at DESC);

-- ChromaDB sync monitoring
CREATE INDEX IF NOT EXISTS idx_chroma_sync_status
ON chroma_sync_status(sync_status, last_synced);

-- ================================
-- Simple Functions and Triggers
-- ================================

-- Update timestamp function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $func$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END
$func$ LANGUAGE plpgsql;

-- Apply to memories table
DROP TRIGGER IF EXISTS update_memories_updated_at ON memories;
CREATE TRIGGER update_memories_updated_at
    BEFORE UPDATE ON memories
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Apply to users table
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ================================
-- Constraints and Validations
-- ================================

-- Ensure memory connections don't create self-references
ALTER TABLE memory_connections
ADD CONSTRAINT IF NOT EXISTS check_no_self_reference
CHECK (source_memory_id != target_memory_id);

-- Ensure conversation processing time is reasonable
ALTER TABLE conversations
ADD CONSTRAINT IF NOT EXISTS check_reasonable_processing_time
CHECK (processing_time_ms >= 0 AND processing_time_ms < 300000);

-- Ensure RAG metrics are reasonable
ALTER TABLE rag_metrics
ADD CONSTRAINT IF NOT EXISTS check_reasonable_response_time
CHECK (response_time_ms >= 0 AND response_time_ms < 60000);

-- Ensure sync retry count is reasonable
ALTER TABLE chroma_sync_status
ADD CONSTRAINT IF NOT EXISTS check_reasonable_retry_count
CHECK (retry_count >= 0 AND retry_count <= 10);