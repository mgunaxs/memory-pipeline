-- Create all tables for the memory pipeline
-- Run this after creating types

SET search_path TO memory_pipeline, public;

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
    metadata JSONB DEFAULT '{}'::jsonb,
    entities TEXT[] DEFAULT '{}',
    valid_contexts TEXT[] DEFAULT '{}',
    invalid_contexts TEXT[] DEFAULT '{}',
    extracted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    last_accessed TIMESTAMP WITH TIME ZONE,
    access_count INTEGER DEFAULT 0,
    source_message_id VARCHAR(255),
    source_conversation_id BIGINT,
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

SELECT 'All tables created successfully' as status;