-- Create performance indexes for the memory pipeline
-- Run this after creating all tables

SET search_path TO memory_pipeline, public;

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

SELECT 'All indexes created successfully' as status;