-- Create constraints and validations for the memory pipeline
-- Run this after creating all tables and triggers

SET search_path TO memory_pipeline, public;

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

SELECT 'Constraints created successfully' as status;