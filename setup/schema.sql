-- Memory Pipeline Production Database Schema
-- PostgreSQL 13+ compatible
-- Optimized for high-performance memory retrieval and storage

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm"; -- For full-text search similarity
CREATE EXTENSION IF NOT EXISTS "btree_gin"; -- For compound indexes

-- Users table - Core user profiles and preferences
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(255) PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    settings JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN DEFAULT true,
    total_memories INTEGER DEFAULT 0,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add comments for documentation
COMMENT ON TABLE users IS 'User profiles and preferences for memory management';
COMMENT ON COLUMN users.settings IS 'User preferences stored as JSONB for flexibility';
COMMENT ON COLUMN users.total_memories IS 'Cached count of active memories for performance';

-- Memory categories enum for type safety
CREATE TYPE memory_type AS ENUM ('fact', 'preference', 'event', 'routine', 'emotion');
CREATE TYPE memory_category AS ENUM ('work', 'food', 'health', 'social', 'schedule', 'entertainment', 'emotional');
CREATE TYPE temporal_relevance AS ENUM ('permanent', 'yearly', 'monthly', 'weekly', 'daily', 'temporary');

-- Memories table - Core memory storage with advanced indexing
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
    is_active BOOLEAN DEFAULT true,

    -- Full-text search column
    content_search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- Add table comments
COMMENT ON TABLE memories IS 'Core memory storage with advanced indexing for high-performance retrieval';
COMMENT ON COLUMN memories.metadata IS 'Flexible JSONB storage for memory metadata and embeddings info';
COMMENT ON COLUMN memories.entities IS 'Extracted entities as PostgreSQL array for efficient querying';
COMMENT ON COLUMN memories.content_search_vector IS 'Full-text search vector for PostgreSQL native search';

-- Conversations table - Track conversation history
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

COMMENT ON TABLE conversations IS 'Conversation history for tracking memory extraction context';

-- Memory connections - Track relationships between memories
CREATE TABLE IF NOT EXISTS memory_connections (
    id BIGSERIAL PRIMARY KEY,
    source_memory_id UUID NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
    target_memory_id UUID NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
    connection_type VARCHAR(50) NOT NULL, -- 'similar', 'contradicts', 'updates', 'references'
    strength DECIMAL(3,2) CHECK (strength >= 0 AND strength <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(source_memory_id, target_memory_id, connection_type)
);

COMMENT ON TABLE memory_connections IS 'Track relationships and connections between memories';

-- Memory access log - Track retrieval patterns for optimization
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

-- Partition by date for performance (monthly partitions)
-- This would be set up in the database_setup.py script

COMMENT ON TABLE memory_access_log IS 'Track memory access patterns for analytics and optimization';

-- RAG metrics - Track RAG performance and quality
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
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    date_bucket DATE GENERATED ALWAYS AS (DATE(created_at)) STORED -- For efficient daily aggregation
);

COMMENT ON TABLE rag_metrics IS 'Track RAG system performance and quality metrics';

-- ChromaDB sync status - Track vector database synchronization
CREATE TABLE IF NOT EXISTS chroma_sync_status (
    memory_id UUID PRIMARY KEY REFERENCES memories(memory_id) ON DELETE CASCADE,
    chroma_id VARCHAR(255),
    collection_name VARCHAR(255),
    embedding_model VARCHAR(100),
    last_synced TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    sync_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'synced', 'failed', 'deleted'
    error_message TEXT,
    retry_count INTEGER DEFAULT 0
);

COMMENT ON TABLE chroma_sync_status IS 'Track synchronization status with ChromaDB cloud';

-- ================================
-- PERFORMANCE INDEXES
-- ================================

-- Critical indexes for memory retrieval performance

-- Primary user-based queries (most common)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_user_active
ON memories(user_id, is_active) WHERE is_active = true;

-- Context filtering (morning/evening routines)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_user_category_type
ON memories(user_id, category, memory_type) WHERE is_active = true;

-- Temporal queries (recent memories, expiration)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_user_created
ON memories(user_id, extracted_at DESC) WHERE is_active = true;

-- Importance-based retrieval
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_user_importance
ON memories(user_id, importance_score DESC, extracted_at DESC) WHERE is_active = true;

-- Expiration management
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_expires_at
ON memories(expires_at) WHERE expires_at IS NOT NULL AND is_active = true;

-- Full-text search index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_search_vector
ON memories USING GIN(content_search_vector);

-- Entity-based searches
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_entities
ON memories USING GIN(entities);

-- Context arrays for smart filtering
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_valid_contexts
ON memories USING GIN(valid_contexts);

-- JSONB metadata queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_metadata
ON memories USING GIN(metadata);

-- User activity and statistics
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_last_activity
ON users(last_activity DESC) WHERE is_active = true;

-- Conversation tracking
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_user_created
ON conversations(user_id, created_at DESC);

-- Memory access analytics
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_access_log_user_date
ON memory_access_log(user_id, accessed_at DESC);

-- RAG metrics aggregation
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rag_metrics_date_bucket
ON rag_metrics(date_bucket, user_id);

-- ChromaDB sync monitoring
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chroma_sync_status
ON chroma_sync_status(sync_status, last_synced);

-- ================================
-- TRIGGERS AND FUNCTIONS
-- ================================

-- Update updated_at timestamp automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply to memories table
CREATE TRIGGER update_memories_updated_at
    BEFORE UPDATE ON memories
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Apply to users table
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Update user memory count when memories change
CREATE OR REPLACE FUNCTION update_user_memory_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' AND NEW.is_active = true THEN
        UPDATE users SET total_memories = total_memories + 1 WHERE user_id = NEW.user_id;
    ELSIF TG_OP = 'UPDATE' THEN
        IF OLD.is_active = true AND NEW.is_active = false THEN
            UPDATE users SET total_memories = total_memories - 1 WHERE user_id = NEW.user_id;
        ELSIF OLD.is_active = false AND NEW.is_active = true THEN
            UPDATE users SET total_memories = total_memories + 1 WHERE user_id = NEW.user_id;
        END IF;
    ELSIF TG_OP = 'DELETE' AND OLD.is_active = true THEN
        UPDATE users SET total_memories = total_memories - 1 WHERE user_id = OLD.user_id;
    END IF;

    RETURN COALESCE(NEW, OLD);
END;
$$ language 'plpgsql';

CREATE TRIGGER update_user_memory_count_trigger
    AFTER INSERT OR UPDATE OR DELETE ON memories
    FOR EACH ROW
    EXECUTE FUNCTION update_user_memory_count();

-- Update access count when memory is accessed
CREATE OR REPLACE FUNCTION update_memory_access()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE memories
    SET access_count = access_count + 1,
        last_accessed = NOW()
    WHERE memory_id = NEW.memory_id;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_memory_access_trigger
    AFTER INSERT ON memory_access_log
    FOR EACH ROW
    EXECUTE FUNCTION update_memory_access();

-- ================================
-- CONSTRAINTS AND VALIDATIONS
-- ================================

-- Ensure memory connections don't create self-references
ALTER TABLE memory_connections
ADD CONSTRAINT check_no_self_reference
CHECK (source_memory_id != target_memory_id);

-- Ensure conversation processing time is reasonable
ALTER TABLE conversations
ADD CONSTRAINT check_reasonable_processing_time
CHECK (processing_time_ms >= 0 AND processing_time_ms < 300000); -- Max 5 minutes

-- Ensure RAG metrics are reasonable
ALTER TABLE rag_metrics
ADD CONSTRAINT check_reasonable_response_time
CHECK (response_time_ms >= 0 AND response_time_ms < 60000); -- Max 1 minute

-- Ensure sync retry count is reasonable
ALTER TABLE chroma_sync_status
ADD CONSTRAINT check_reasonable_retry_count
CHECK (retry_count >= 0 AND retry_count <= 10);

-- ================================
-- VIEWS FOR COMMON QUERIES
-- ================================

-- Active memories with user info
CREATE VIEW active_memories_view AS
SELECT
    m.*,
    u.settings as user_settings,
    u.last_activity as user_last_activity
FROM memories m
JOIN users u ON m.user_id = u.user_id
WHERE m.is_active = true
    AND u.is_active = true
    AND (m.expires_at IS NULL OR m.expires_at > NOW());

-- Memory statistics by user
CREATE VIEW user_memory_stats AS
SELECT
    u.user_id,
    u.total_memories,
    COUNT(CASE WHEN m.category = 'work' THEN 1 END) as work_memories,
    COUNT(CASE WHEN m.category = 'food' THEN 1 END) as food_memories,
    COUNT(CASE WHEN m.category = 'health' THEN 1 END) as health_memories,
    COUNT(CASE WHEN m.category = 'social' THEN 1 END) as social_memories,
    COUNT(CASE WHEN m.category = 'schedule' THEN 1 END) as schedule_memories,
    COUNT(CASE WHEN m.category = 'entertainment' THEN 1 END) as entertainment_memories,
    COUNT(CASE WHEN m.category = 'emotional' THEN 1 END) as emotional_memories,
    AVG(m.importance_score) as avg_importance,
    AVG(m.confidence_score) as avg_confidence,
    MAX(m.extracted_at) as latest_memory,
    COUNT(*) as active_memory_count
FROM users u
LEFT JOIN memories m ON u.user_id = m.user_id AND m.is_active = true
WHERE u.is_active = true
GROUP BY u.user_id, u.total_memories;

-- ================================
-- PERFORMANCE OPTIMIZATION
-- ================================

-- Set reasonable autovacuum settings for high-write tables
ALTER TABLE memory_access_log SET (
    autovacuum_vacuum_scale_factor = 0.1,
    autovacuum_analyze_scale_factor = 0.05
);

ALTER TABLE rag_metrics SET (
    autovacuum_vacuum_scale_factor = 0.1,
    autovacuum_analyze_scale_factor = 0.05
);

-- Set fill factor for tables with frequent updates
ALTER TABLE memories SET (fillfactor = 85);
ALTER TABLE users SET (fillfactor = 85);

-- ================================
-- SECURITY SETTINGS
-- ================================

-- Create application role with limited permissions
CREATE ROLE memory_app_role;
GRANT CONNECT ON DATABASE memory_pipeline TO memory_app_role;
GRANT USAGE ON SCHEMA public TO memory_app_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO memory_app_role;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO memory_app_role;

-- Row-level security for multi-tenant access
ALTER TABLE memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory_access_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_metrics ENABLE ROW LEVEL SECURITY;

-- Example RLS policy (would be customized based on authentication)
-- CREATE POLICY user_isolation ON memories FOR ALL TO memory_app_role USING (user_id = current_setting('app.current_user'));

-- ================================
-- FINAL NOTES
-- ================================

-- This schema is designed for:
-- 1. High-performance memory retrieval (<500ms)
-- 2. Efficient context-based filtering
-- 3. Full-text search capabilities
-- 4. Proper analytics and monitoring
-- 5. ChromaDB cloud synchronization
-- 6. Multi-tenant security
-- 7. Automatic maintenance and optimization

-- For production deployment:
-- 1. Review and adjust autovacuum settings based on workload
-- 2. Set up proper monitoring for slow queries
-- 3. Configure connection pooling (recommended: 20-40 connections)
-- 4. Enable statement timeout (recommended: 30s)
-- 5. Set up regular ANALYZE for query plan optimization
-- 6. Configure WAL archiving for point-in-time recovery