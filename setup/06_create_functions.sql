-- Create functions and triggers for the memory pipeline
-- Run this after creating all tables and indexes

SET search_path TO memory_pipeline, public;

-- Update timestamp function
CREATE OR REPLACE FUNCTION memory_pipeline.update_updated_at_column()
RETURNS TRIGGER AS $func$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END
$func$ LANGUAGE plpgsql;

SELECT 'Functions created successfully' as status;