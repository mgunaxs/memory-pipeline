-- Drop existing schema (if exists) for clean setup
-- Run this first to ensure clean state

DROP SCHEMA IF EXISTS memory_pipeline CASCADE;

-- Note: This will drop all tables, functions, and data in the memory_pipeline schema
-- Only run this if you want a complete fresh start

SELECT 'Schema memory_pipeline dropped (if it existed)' as status;