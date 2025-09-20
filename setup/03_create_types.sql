-- Create custom types for the memory pipeline
-- Run this after creating the schema

SET search_path TO memory_pipeline, public;

-- Memory types and categories
CREATE TYPE memory_pipeline.memory_type AS ENUM ('fact', 'preference', 'event', 'routine', 'emotion');
CREATE TYPE memory_pipeline.memory_category AS ENUM ('work', 'food', 'health', 'social', 'schedule', 'entertainment', 'emotional');
CREATE TYPE memory_pipeline.temporal_relevance AS ENUM ('permanent', 'yearly', 'monthly', 'weekly', 'daily', 'temporary');

SELECT 'Custom types created successfully' as status;