#!/usr/bin/env python3
"""
SQLite to PostgreSQL Migration Script
Safely migrates all data from SQLite to production PostgreSQL database.
"""

import os
import sys
import time
import logging
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import psycopg2
from psycopg2.extras import execute_batch
import asyncpg

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
        logging.FileHandler('scripts/migration.log')
    ]
)
logger = logging.getLogger(__name__)


class SQLiteToPostgreSQLMigrator:
    """
    Handles migration from SQLite to PostgreSQL with data integrity checks.
    """

    def __init__(self, sqlite_path: str, postgresql_url: str):
        """
        Initialize migrator.

        Args:
            sqlite_path: Path to SQLite database
            postgresql_url: PostgreSQL connection URL
        """
        self.sqlite_path = sqlite_path
        self.postgresql_url = postgresql_url
        self.migration_stats = {
            'start_time': time.time(),
            'tables_migrated': 0,
            'records_migrated': 0,
            'errors': [],
            'table_stats': {}
        }

    def check_sqlite_database(self) -> bool:
        """
        Check if SQLite database exists and is accessible.

        Returns:
            bool: True if SQLite database is accessible
        """
        try:
            if not os.path.exists(self.sqlite_path):
                logger.warning(f"SQLite database not found: {self.sqlite_path}")
                return False

            # Test connection
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()

            # Check for memory pipeline tables
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN ('users', 'memories', 'conversations');
            """)

            tables = cursor.fetchall()
            cursor.close()
            conn.close()

            if not tables:
                logger.warning("No memory pipeline tables found in SQLite database")
                return False

            logger.info(f"SQLite database found with {len(tables)} tables")
            return True

        except Exception as e:
            logger.error(f"Failed to check SQLite database: {e}")
            return False

    def check_postgresql_database(self) -> bool:
        """
        Check if PostgreSQL database is accessible.

        Returns:
            bool: True if PostgreSQL database is accessible
        """
        try:
            conn = psycopg2.connect(self.postgresql_url)
            cursor = conn.cursor()

            # Test basic query
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]

            logger.info(f"PostgreSQL connection successful: {version[:50]}...")

            cursor.close()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False

    def get_sqlite_schema(self) -> Dict[str, List[str]]:
        """
        Get SQLite table schemas.

        Returns:
            Dict mapping table names to column definitions
        """
        schemas = {}

        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()

            # Get all tables
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%';
            """)

            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                cursor.execute(f"PRAGMA table_info({table});")
                columns = cursor.fetchall()
                schemas[table] = [
                    {
                        'name': col[1],
                        'type': col[2],
                        'not_null': col[3],
                        'default': col[4],
                        'primary_key': col[5]
                    }
                    for col in columns
                ]

            cursor.close()
            conn.close()

            logger.info(f"Retrieved schema for {len(schemas)} tables")
            return schemas

        except Exception as e:
            logger.error(f"Failed to get SQLite schema: {e}")
            return {}

    def migrate_table_data(self, table_name: str, batch_size: int = 1000) -> bool:
        """
        Migrate data from one table.

        Args:
            table_name: Name of table to migrate
            batch_size: Number of records per batch

        Returns:
            bool: True if migration successful
        """
        try:
            # Connect to both databases
            sqlite_conn = sqlite3.connect(self.sqlite_path)
            sqlite_conn.row_factory = sqlite3.Row  # Enable column access by name

            pg_conn = psycopg2.connect(self.postgresql_url)
            pg_cursor = pg_conn.cursor()

            # Get SQLite data
            sqlite_cursor = sqlite_conn.cursor()
            sqlite_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_records = sqlite_cursor.fetchone()[0]

            if total_records == 0:
                logger.info(f"Table {table_name} is empty, skipping")
                sqlite_conn.close()
                pg_conn.close()
                return True

            logger.info(f"Migrating {total_records} records from {table_name}")

            # Get all data from SQLite
            sqlite_cursor.execute(f"SELECT * FROM {table_name}")

            migrated_count = 0
            batch_data = []

            while True:
                rows = sqlite_cursor.fetchmany(batch_size)
                if not rows:
                    break

                # Convert SQLite rows to dictionaries
                for row in rows:
                    row_dict = dict(row)

                    # Handle special data type conversions
                    row_dict = self.convert_data_types(table_name, row_dict)
                    batch_data.append(row_dict)

                # Insert batch into PostgreSQL
                if batch_data:
                    self.insert_batch_to_postgresql(pg_cursor, table_name, batch_data)
                    pg_conn.commit()

                    migrated_count += len(batch_data)
                    batch_data = []

                    logger.debug(f"Migrated {migrated_count}/{total_records} records from {table_name}")

            # Final batch
            if batch_data:
                self.insert_batch_to_postgresql(pg_cursor, table_name, batch_data)
                pg_conn.commit()
                migrated_count += len(batch_data)

            # Update statistics
            self.migration_stats['table_stats'][table_name] = {
                'total_records': total_records,
                'migrated_records': migrated_count,
                'success': migrated_count == total_records
            }

            sqlite_conn.close()
            pg_conn.close()

            logger.info(f"Successfully migrated {migrated_count} records from {table_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to migrate table {table_name}: {e}")
            self.migration_stats['errors'].append(f"Table {table_name}: {e}")
            return False

    def convert_data_types(self, table_name: str, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert SQLite data types to PostgreSQL compatible formats.

        Args:
            table_name: Name of the table
            row_dict: Row data dictionary

        Returns:
            Converted row dictionary
        """
        converted = row_dict.copy()

        # Handle JSON fields
        if table_name == 'memories' and 'metadata_' in converted:
            if isinstance(converted['metadata_'], str):
                try:
                    converted['metadata'] = json.loads(converted['metadata_'])
                except (json.JSONDecodeError, TypeError):
                    converted['metadata'] = {}
            else:
                converted['metadata'] = converted.get('metadata_', {})

            # Remove the SQLite field name
            if 'metadata_' in converted:
                del converted['metadata_']

        # Handle timestamp fields
        timestamp_fields = ['created_at', 'updated_at', 'extracted_at', 'expires_at', 'last_accessed']
        for field in timestamp_fields:
            if field in converted and converted[field] is not None:
                # Convert to PostgreSQL timestamp format if needed
                if isinstance(converted[field], str):
                    try:
                        # Try parsing different datetime formats
                        dt = datetime.fromisoformat(converted[field].replace('Z', '+00:00'))
                        converted[field] = dt
                    except ValueError:
                        # Keep as string for PostgreSQL to handle
                        pass

        # Handle boolean fields
        boolean_fields = ['is_active']
        for field in boolean_fields:
            if field in converted and converted[field] is not None:
                converted[field] = bool(converted[field])

        # Handle array fields for PostgreSQL
        if table_name == 'memories':
            array_fields = ['entities', 'valid_contexts', 'invalid_contexts']
            for field in array_fields:
                if field in converted:
                    if isinstance(converted[field], str):
                        try:
                            converted[field] = json.loads(converted[field])
                        except (json.JSONDecodeError, TypeError):
                            converted[field] = []
                    elif converted[field] is None:
                        converted[field] = []

        return converted

    def insert_batch_to_postgresql(self, cursor, table_name: str, batch_data: List[Dict[str, Any]]):
        """
        Insert a batch of data into PostgreSQL.

        Args:
            cursor: PostgreSQL cursor
            table_name: Name of target table
            batch_data: List of row dictionaries
        """
        if not batch_data:
            return

        # Get column names from first row
        columns = list(batch_data[0].keys())

        # Create INSERT statement
        placeholders = ', '.join(['%s'] * len(columns))
        columns_str = ', '.join(columns)

        insert_sql = f"""
            INSERT INTO {table_name} ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT DO NOTHING
        """

        # Convert dictionaries to tuples in correct order
        values_list = [
            tuple(row.get(col) for col in columns)
            for row in batch_data
        ]

        # Execute batch insert
        execute_batch(cursor, insert_sql, values_list, page_size=100)

    def verify_migration(self) -> Dict[str, Any]:
        """
        Verify that migration was successful by comparing record counts.

        Returns:
            Verification results
        """
        verification_results = {}

        try:
            # Connect to both databases
            sqlite_conn = sqlite3.connect(self.sqlite_path)
            sqlite_cursor = sqlite_conn.cursor()

            pg_conn = psycopg2.connect(self.postgresql_url)
            pg_cursor = pg_conn.cursor()

            # Get list of tables
            sqlite_cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%';
            """)
            tables = [row[0] for row in sqlite_cursor.fetchall()]

            for table in tables:
                # Count records in SQLite
                sqlite_cursor.execute(f"SELECT COUNT(*) FROM {table}")
                sqlite_count = sqlite_cursor.fetchone()[0]

                # Count records in PostgreSQL
                try:
                    pg_cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    pg_count = pg_cursor.fetchone()[0]
                except psycopg2.Error:
                    pg_count = 0

                verification_results[table] = {
                    'sqlite_count': sqlite_count,
                    'postgresql_count': pg_count,
                    'match': sqlite_count == pg_count
                }

            sqlite_conn.close()
            pg_conn.close()

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            verification_results['error'] = str(e)

        return verification_results

    def create_migration_report(self) -> str:
        """
        Create a detailed migration report.

        Returns:
            Formatted migration report
        """
        migration_time = time.time() - self.migration_stats['start_time']
        verification = self.verify_migration()

        report = f"""
==========================================
SQLITE TO POSTGRESQL MIGRATION REPORT
==========================================

Migration Time: {migration_time:.2f} seconds
Tables Migrated: {self.migration_stats['tables_migrated']}
Total Records: {self.migration_stats['records_migrated']}
Errors: {len(self.migration_stats['errors'])}

SOURCE DATABASE: {self.sqlite_path}
TARGET DATABASE: {self.postgresql_url.split('@')[1] if '@' in self.postgresql_url else 'localhost'}

"""

        # Table-by-table results
        report += "TABLE MIGRATION RESULTS:\n"
        for table, stats in self.migration_stats['table_stats'].items():
            status = "✓" if stats['success'] else "✗"
            report += f"  {status} {table}: {stats['migrated_records']} records\n"

        # Verification results
        if 'error' not in verification:
            report += "\nVERIFICATION RESULTS:\n"
            for table, results in verification.items():
                match_status = "✓" if results['match'] else "✗"
                report += f"  {match_status} {table}: SQLite({results['sqlite_count']}) -> PostgreSQL({results['postgresql_count']})\n"

        # Errors
        if self.migration_stats['errors']:
            report += "\nERRORS ENCOUNTERED:\n"
            for i, error in enumerate(self.migration_stats['errors'], 1):
                report += f"  {i}. {error}\n"

        report += "\n==========================================\n"

        return report

    def run_migration(self) -> bool:
        """
        Run the complete migration process.

        Returns:
            bool: True if migration successful
        """
        logger.info("Starting SQLite to PostgreSQL migration...")

        try:
            # Step 1: Check databases
            if not self.check_sqlite_database():
                logger.error("SQLite database check failed")
                return False

            if not self.check_postgresql_database():
                logger.error("PostgreSQL database check failed")
                return False

            # Step 2: Get SQLite schema
            schemas = self.get_sqlite_schema()
            if not schemas:
                logger.error("Failed to get SQLite schema")
                return False

            # Step 3: Migrate each table
            migration_order = ['users', 'conversations', 'memories', 'memory_connections']

            for table in migration_order:
                if table in schemas:
                    logger.info(f"Migrating table: {table}")
                    success = self.migrate_table_data(table)
                    if success:
                        self.migration_stats['tables_migrated'] += 1
                        self.migration_stats['records_migrated'] += self.migration_stats['table_stats'].get(table, {}).get('migrated_records', 0)
                    else:
                        logger.error(f"Failed to migrate table: {table}")

            # Step 4: Migrate any remaining tables
            for table in schemas:
                if table not in migration_order:
                    logger.info(f"Migrating additional table: {table}")
                    success = self.migrate_table_data(table)
                    if success:
                        self.migration_stats['tables_migrated'] += 1
                        self.migration_stats['records_migrated'] += self.migration_stats['table_stats'].get(table, {}).get('migrated_records', 0)

            # Step 5: Generate and save report
            report = self.create_migration_report()
            print(report)

            # Save report to file
            with open('scripts/migration_report.txt', 'w') as f:
                f.write(report)

            # Step 6: Final verification
            verification = self.verify_migration()
            all_verified = all(
                result.get('match', False)
                for result in verification.values()
                if isinstance(result, dict)
            )

            if all_verified and not self.migration_stats['errors']:
                logger.info("Migration completed successfully!")
                return True
            else:
                logger.warning("Migration completed with issues - check the report")
                return False

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            self.migration_stats['errors'].append(f"Migration failed: {e}")
            return False


def main():
    """Main migration function."""
    print("Memory Pipeline - SQLite to PostgreSQL Migration")
    print("=" * 50)

    # Check environment
    if not settings.database_url:
        print("ERROR: DATABASE_URL environment variable not set")
        print("Please set DATABASE_URL=postgresql://user:password@host:port/database")
        return False

    # Default SQLite path
    sqlite_path = "data/memory.db"
    if not os.path.exists(sqlite_path):
        sqlite_path = "data/sqlite/memories.db"

    if not os.path.exists(sqlite_path):
        print(f"No SQLite database found at {sqlite_path}")
        print("Migration not needed - starting fresh with PostgreSQL")
        return True

    # Ask for confirmation
    print(f"\nSource SQLite: {sqlite_path}")
    print(f"Target PostgreSQL: {settings.database_url.split('@')[1] if '@' in settings.database_url else 'localhost'}")

    confirm = input("\nProceed with migration? (y/N): ").lower().strip()
    if confirm != 'y':
        print("Migration cancelled")
        return False

    # Run migration
    migrator = SQLiteToPostgreSQLMigrator(sqlite_path, settings.database_url)
    success = migrator.run_migration()

    if success:
        print("\n✓ Migration completed successfully!")
        print("Next steps:")
        print("1. Test the application with PostgreSQL")
        print("2. Run health checks: python -m pytest tests/test_health.py")
        print("3. If everything works, you can safely remove the SQLite database")
    else:
        print("\n✗ Migration failed!")
        print("Check the migration report and logs for details")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)