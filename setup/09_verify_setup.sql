-- Verify schema setup is complete and correct
-- Run this last to confirm everything was created properly

SET search_path TO memory_pipeline, public;

-- Verify schema creation
SELECT 'Schema verification:' as check_type;
SELECT schemaname, tablename, tableowner
FROM pg_tables
WHERE schemaname = 'memory_pipeline'
ORDER BY tablename;

-- Verify indexes
SELECT 'Index verification:' as check_type;
SELECT schemaname, indexname, tablename
FROM pg_indexes
WHERE schemaname = 'memory_pipeline'
ORDER BY tablename, indexname;

-- Verify types
SELECT 'Type verification:' as check_type;
SELECT typname
FROM pg_type t
JOIN pg_namespace n ON t.typnamespace = n.oid
WHERE n.nspname = 'memory_pipeline'
ORDER BY typname;

-- Verify functions
SELECT 'Function verification:' as check_type;
SELECT proname
FROM pg_proc p
JOIN pg_namespace n ON p.pronamespace = n.oid
WHERE n.nspname = 'memory_pipeline'
ORDER BY proname;

-- Show current search path
SELECT 'Current search path:' as check_type;
SHOW search_path;

SELECT 'Database setup verification completed successfully!' as final_status;