#!/usr/bin/env python3
"""
Production Database Setup Script
Creates PostgreSQL tables, indexes, and configurations for Memory Pipeline.
Idempotent and safe to run multiple times.
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import asyncpg
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2 import sql

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('setup/database_setup.log')
    ]
)
logger = logging.getLogger(__name__)


class DatabaseSetup:
    """
    Production-ready PostgreSQL database setup with comprehensive error handling.
    """

    def __init__(self):
        """Initialize database setup with connection parameters."""
        self.db_url = settings.database_url
        self.setup_stats = {
            'start_time': time.time(),
            'tables_created': 0,
            'indexes_created': 0,
            'triggers_created': 0,
            'errors': []
        }

    def parse_database_url(self) -> Dict[str, str]:
        """
        Parse PostgreSQL database URL into connection parameters.

        Returns:
            Dict with connection parameters
        """
        if not self.db_url.startswith('postgresql://'):
            raise ValueError("DATABASE_URL must start with 'postgresql://'")

        # Remove postgresql:// prefix
        url_parts = self.db_url.replace('postgresql://', '').split('/')
        connection_part = url_parts[0]
        database = url_parts[1] if len(url_parts) > 1 else 'postgres'

        # Parse user:password@host:port
        if '@' in connection_part:
            auth_part, host_part = connection_part.split('@')
            if ':' in auth_part:
                user, password = auth_part.split(':')
            else:
                user, password = auth_part, ''
        else:
            user, password = 'postgres', ''
            host_part = connection_part

        if ':' in host_part:
            host, port = host_part.split(':')
        else:
            host, port = host_part, '5432'

        return {
            'host': host,
            'port': int(port),
            'user': user,
            'password': password,
            'database': database
        }

    def test_connection(self) -> bool:
        """
        Test database connection and basic functionality.

        Returns:
            True if connection successful
        """
        try:
            conn_params = self.parse_database_url()
            logger.info(f"Testing connection to {conn_params['host']}:{conn_params['port']}")

            # Test connection
            conn = psycopg2.connect(**conn_params)
            cursor = conn.cursor()

            # Test basic query
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            logger.info(f"Connected to PostgreSQL: {version}")

            # Test database creation privileges
            cursor.execute("SELECT current_user, session_user;")
            user_info = cursor.fetchone()
            logger.info(f"Connected as user: {user_info[0]}")

            cursor.close()
            conn.close()

            return True

        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            self.setup_stats['errors'].append(f"Connection test failed: {e}")
            return False

    def create_database_if_not_exists(self) -> bool:
        """
        Create the database if it doesn't exist.

        Returns:
            True if database exists or was created successfully
        """
        try:
            conn_params = self.parse_database_url()
            target_db = conn_params['database']

            # Connect to postgres database first
            postgres_params = conn_params.copy()
            postgres_params['database'] = 'postgres'

            conn = psycopg2.connect(**postgres_params)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s;",
                (target_db,)
            )

            if cursor.fetchone():
                logger.info(f"Database '{target_db}' already exists")
            else:
                # Create database
                cursor.execute(
                    sql.SQL("CREATE DATABASE {} WITH ENCODING 'UTF8'").format(
                        sql.Identifier(target_db)
                    )
                )
                logger.info(f"Created database '{target_db}'")

            cursor.close()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Database creation failed: {e}")
            self.setup_stats['errors'].append(f"Database creation failed: {e}")
            return False

    def execute_sql_file(self, filepath: str) -> bool:
        """
        Execute SQL commands from a file.

        Args:
            filepath: Path to SQL file

        Returns:
            True if execution successful
        """
        try:
            conn_params = self.parse_database_url()
            conn = psycopg2.connect(**conn_params)
            cursor = conn.cursor()

            # Read and execute SQL file
            with open(filepath, 'r', encoding='utf-8') as f:
                sql_content = f.read()

            # Split by statement and execute
            statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]

            for i, statement in enumerate(statements):
                try:
                    if statement.strip():
                        cursor.execute(statement)
                        conn.commit()

                        # Count different types of operations
                        statement_lower = statement.lower()
                        if 'create table' in statement_lower:
                            self.setup_stats['tables_created'] += 1
                        elif 'create index' in statement_lower:
                            self.setup_stats['indexes_created'] += 1
                        elif 'create trigger' in statement_lower:
                            self.setup_stats['triggers_created'] += 1

                except psycopg2.Error as e:
                    if 'already exists' in str(e).lower():
                        logger.debug(f"Object already exists (statement {i+1}): {e}")
                    else:
                        logger.warning(f"Error in statement {i+1}: {e}")
                        self.setup_stats['errors'].append(f"Statement {i+1}: {e}")

            cursor.close()
            conn.close()

            logger.info(f"Successfully executed SQL file: {filepath}")
            return True

        except Exception as e:
            logger.error(f"SQL file execution failed: {e}")
            self.setup_stats['errors'].append(f"SQL execution failed: {e}")
            return False

    def verify_schema(self) -> Dict[str, Any]:
        """
        Verify that all expected tables and indexes exist.

        Returns:
            Dict with verification results
        """
        verification_results = {
            'tables': {},
            'indexes': {},
            'functions': {},
            'triggers': {}
        }

        try:
            conn_params = self.parse_database_url()
            conn = psycopg2.connect(**conn_params)
            cursor = conn.cursor()

            # Check tables
            expected_tables = [
                'users', 'memories', 'conversations', 'memory_connections',
                'memory_access_log', 'rag_metrics', 'chroma_sync_status'
            ]

            for table in expected_tables:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = %s
                    );
                """, (table,))

                verification_results['tables'][table] = cursor.fetchone()[0]

            # Check critical indexes
            critical_indexes = [
                'idx_memories_user_active',
                'idx_memories_user_category_type',
                'idx_memories_user_created',
                'idx_memories_user_importance',
                'idx_memories_search_vector'
            ]

            for index in critical_indexes:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM pg_indexes
                        WHERE schemaname = 'public'
                        AND indexname = %s
                    );
                """, (index,))

                verification_results['indexes'][index] = cursor.fetchone()[0]

            # Check functions
            expected_functions = [
                'update_updated_at_column',
                'update_user_memory_count',
                'update_memory_access'
            ]

            for function in expected_functions:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM pg_proc
                        WHERE proname = %s
                    );
                """, (function,))

                verification_results['functions'][function] = cursor.fetchone()[0]

            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Schema verification failed: {e}")
            verification_results['error'] = str(e)

        return verification_results

    def run_performance_checks(self) -> Dict[str, Any]:
        """
        Run basic performance checks on the database.

        Returns:
            Dict with performance metrics
        """
        performance_results = {}

        try:
            conn_params = self.parse_database_url()
            conn = psycopg2.connect(**conn_params)
            cursor = conn.cursor()

            # Test basic query performance
            start_time = time.time()
            cursor.execute("SELECT COUNT(*) FROM users;")
            user_count = cursor.fetchone()[0]
            query_time = (time.time() - start_time) * 1000

            performance_results['user_count_query_ms'] = query_time
            performance_results['user_count'] = user_count

            # Test index usage
            cursor.execute("""
                SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read
                FROM pg_stat_user_indexes
                WHERE schemaname = 'public'
                ORDER BY idx_scan DESC;
            """)

            index_stats = cursor.fetchall()
            performance_results['index_usage'] = [
                {
                    'table': row[1],
                    'index': row[2],
                    'scans': row[3],
                    'tuples_read': row[4]
                }
                for row in index_stats[:10]  # Top 10 most used indexes
            ]

            # Test connection pool compatibility
            start_time = time.time()
            for _ in range(5):
                test_conn = psycopg2.connect(**conn_params)
                test_cursor = test_conn.cursor()
                test_cursor.execute("SELECT 1;")
                test_cursor.close()
                test_conn.close()

            connection_test_time = (time.time() - start_time) * 1000
            performance_results['connection_pool_test_ms'] = connection_test_time

            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Performance checks failed: {e}")
            performance_results['error'] = str(e)

        return performance_results

    def setup_monitoring(self) -> bool:
        """
        Set up basic monitoring and statistics collection.

        Returns:
            True if setup successful
        """
        try:
            conn_params = self.parse_database_url()
            conn = psycopg2.connect(**conn_params)
            cursor = conn.cursor()

            # Enable query statistics collection
            cursor.execute("SELECT name, setting FROM pg_settings WHERE name LIKE 'track_%';")
            settings_before = dict(cursor.fetchall())

            # Update settings for better monitoring
            monitoring_settings = [
                "ALTER SYSTEM SET track_activities = on;",
                "ALTER SYSTEM SET track_counts = on;",
                "ALTER SYSTEM SET track_io_timing = on;",
                "ALTER SYSTEM SET track_functions = 'all';",
                "ALTER SYSTEM SET log_statement = 'ddl';",
                "ALTER SYSTEM SET log_min_duration_statement = 1000;"  # Log slow queries
            ]

            for setting in monitoring_settings:
                try:
                    cursor.execute(setting)
                    conn.commit()
                except psycopg2.Error as e:
                    logger.warning(f"Could not set monitoring setting: {e}")

            logger.info("Monitoring settings configured (restart required for some settings)")

            cursor.close()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Monitoring setup failed: {e}")
            return False

    def create_application_user(self) -> bool:
        """
        Create application-specific database user with proper permissions.

        Returns:
            True if user creation successful
        """
        try:
            conn_params = self.parse_database_url()
            conn = psycopg2.connect(**conn_params)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            app_user = os.getenv('DB_APP_USER', 'memory_app')
            app_password = os.getenv('DB_APP_PASSWORD', 'change_me_in_production')

            # Check if user exists
            cursor.execute(
                "SELECT 1 FROM pg_roles WHERE rolname = %s;",
                (app_user,)
            )

            if cursor.fetchone():
                logger.info(f"Application user '{app_user}' already exists")
            else:
                # Create user
                cursor.execute(
                    sql.SQL("CREATE ROLE {} WITH LOGIN PASSWORD %s;").format(
                        sql.Identifier(app_user)
                    ),
                    (app_password,)
                )
                logger.info(f"Created application user '{app_user}'")

            # Grant permissions
            cursor.execute(
                sql.SQL("GRANT memory_app_role TO {};").format(
                    sql.Identifier(app_user)
                )
            )

            cursor.close()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Application user creation failed: {e}")
            return False

    def generate_setup_report(self) -> str:
        """
        Generate a comprehensive setup report.

        Returns:
            Formatted setup report
        """
        setup_time = time.time() - self.setup_stats['start_time']

        report = f"""
==========================================
MEMORY PIPELINE DATABASE SETUP REPORT
==========================================

Setup Time: {setup_time:.2f} seconds
Tables Created: {self.setup_stats['tables_created']}
Indexes Created: {self.setup_stats['indexes_created']}
Triggers Created: {self.setup_stats['triggers_created']}
Errors: {len(self.setup_stats['errors'])}

Database URL: {self.db_url.split('@')[1] if '@' in self.db_url else 'localhost'}

"""

        if self.setup_stats['errors']:
            report += "ERRORS ENCOUNTERED:\n"
            for i, error in enumerate(self.setup_stats['errors'], 1):
                report += f"{i}. {error}\n"
            report += "\n"

        # Add verification results
        verification = self.verify_schema()
        report += "SCHEMA VERIFICATION:\n"

        report += "\nTables:\n"
        for table, exists in verification['tables'].items():
            status = "✓" if exists else "✗"
            report += f"  {status} {table}\n"

        report += "\nCritical Indexes:\n"
        for index, exists in verification['indexes'].items():
            status = "✓" if exists else "✗"
            report += f"  {status} {index}\n"

        report += "\nFunctions:\n"
        for function, exists in verification['functions'].items():
            status = "✓" if exists else "✗"
            report += f"  {status} {function}\n"

        # Add performance results
        performance = self.run_performance_checks()
        if 'error' not in performance:
            report += f"\nPERFORMANCE METRICS:\n"
            report += f"User Count Query: {performance.get('user_count_query_ms', 0):.2f}ms\n"
            report += f"Connection Pool Test: {performance.get('connection_pool_test_ms', 0):.2f}ms\n"
            report += f"Current User Count: {performance.get('user_count', 0)}\n"

        report += "\n==========================================\n"

        return report

    def run_setup(self) -> bool:
        """
        Run the complete database setup process.

        Returns:
            True if setup completed successfully
        """
        logger.info("Starting Memory Pipeline database setup...")

        try:
            # Step 1: Test connection
            if not self.test_connection():
                logger.error("Database connection failed. Setup aborted.")
                return False

            # Step 2: Create database if needed
            if not self.create_database_if_not_exists():
                logger.error("Database creation failed. Setup aborted.")
                return False

            # Step 3: Execute schema
            schema_file = project_root / 'setup' / 'schema.sql'
            if not schema_file.exists():
                logger.error(f"Schema file not found: {schema_file}")
                return False

            if not self.execute_sql_file(str(schema_file)):
                logger.error("Schema creation failed. Setup aborted.")
                return False

            # Step 4: Set up monitoring
            self.setup_monitoring()

            # Step 5: Create application user (optional)
            if os.getenv('CREATE_APP_USER', 'false').lower() == 'true':
                self.create_application_user()

            # Step 6: Generate and display report
            report = self.generate_setup_report()
            print(report)

            # Save report to file
            with open('setup/database_setup_report.txt', 'w') as f:
                f.write(report)

            logger.info("Database setup completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            self.setup_stats['errors'].append(f"Setup failed: {e}")
            return False


async def test_async_connection() -> bool:
    """
    Test async database connectivity for future async operations.

    Returns:
        True if async connection successful
    """
    try:
        # Parse database URL for asyncpg
        db_url = settings.database_url

        # Connect using asyncpg
        conn = await asyncpg.connect(db_url)

        # Test basic query
        version = await conn.fetchval("SELECT version();")
        logger.info(f"Async connection successful: {version[:50]}...")

        await conn.close()
        return True

    except Exception as e:
        logger.error(f"Async connection test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("Memory Pipeline - Production Database Setup")
    print("=" * 50)

    # Check environment
    if not settings.database_url:
        print("ERROR: DATABASE_URL environment variable not set")
        print("Please set DATABASE_URL=postgresql://user:password@host:port/database")
        return False

    # Run setup
    setup = DatabaseSetup()
    success = setup.run_setup()

    if success:
        print("\n✓ Database setup completed successfully!")
        print("Next steps:")
        print("1. Run tests: python -m pytest tests/test_health.py")
        print("2. Start API: python -m uvicorn app.main:app --reload")
        print("3. Check health: curl http://localhost:8000/api/v1/memory/health")

        # Test async connectivity
        print("\nTesting async connectivity...")
        async_success = asyncio.run(test_async_connection())
        if async_success:
            print("✓ Async database connectivity verified")
        else:
            print("⚠ Async connectivity test failed")
    else:
        print("\n✗ Database setup failed!")
        print("Check the logs and fix any errors before proceeding.")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)