-- Create dedicated schema for memory pipeline
-- This avoids using the public schema for better organization

CREATE SCHEMA memory_pipeline;

-- Set search path to use our schema by default
SET search_path TO memory_pipeline, public;

-- Enable required extensions in public schema (standard practice)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" SCHEMA public;

SELECT 'Schema memory_pipeline created successfully' as status;