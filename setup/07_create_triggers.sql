-- Create triggers for the memory pipeline
-- Run this after creating functions

SET search_path TO memory_pipeline, public;

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

SELECT 'Triggers created successfully' as status;