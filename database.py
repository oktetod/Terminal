# Filename: database.py
import asyncpg
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = "postgres://mouse:sP7=hH8+rK3=gL4=pJ8_@nursing-plum-pony-klxzh-postgresql.nursing-plum-pony-klxzh.svc.cluster.local:5432/nursing-plum-pony"

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set!")

# Global connection pool
_pool: Optional[asyncpg.Pool] = None

# ===================================================================
# CONNECTION POOL MANAGEMENT
# ===================================================================
async def get_pool() -> asyncpg.Pool:
    """Get or create the database connection pool"""
    global _pool
    
    if _pool is None or _pool._closed:
        logger.info("Creating new database connection pool...")
        _pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=2,
            max_size=10,
            max_queries=50000,
            max_inactive_connection_lifetime=300,
            command_timeout=60
        )
        logger.info("✓ Database connection pool created successfully.")
    
    return _pool

async def close_pool():
    """Close the database connection pool"""
    global _pool
    
    if _pool and not _pool._closed:
        await _pool.close()
        logger.info("Database connection pool closed.")
        _pool = None

# ===================================================================
# DATABASE INITIALIZATION
# ===================================================================
async def init_db():
    """Initialize database schema - create tables if they don't exist"""
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        try:
            # Create loras table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS loras (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    added_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            
            # Create index on added_at for faster sorting (if needed)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_loras_added_at 
                ON loras(added_at DESC);
            """)
            
            # Get current count
            count = await conn.fetchval("SELECT COUNT(*) FROM loras;")
            
            logger.info(f"✓ Database initialized. 'loras' table ready with {count} entries.")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}", exc_info=True)
            raise

# ===================================================================
# LORA OPERATIONS
# ===================================================================
async def get_loras_from_db() -> List[str]:
    """
    Retrieve all LoRA names from the database.
    Returns a list of LoRA name strings.
    """
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        try:
            records = await conn.fetch("""
                SELECT name 
                FROM loras 
                ORDER BY name ASC;
            """)
            
            lora_names = [record['name'] for record in records]
            logger.debug(f"Retrieved {len(lora_names)} LoRAs from database.")
            return lora_names
            
        except Exception as e:
            logger.error(f"Error fetching LoRAs from database: {e}", exc_info=True)
            return []

async def get_lora_details(lora_name: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific LoRA.
    Returns dict with name, description, added_at, updated_at or None if not found.
    """
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        try:
            record = await conn.fetchrow("""
                SELECT name, description, added_at, updated_at
                FROM loras
                WHERE name = $1;
            """, lora_name)
            
            if record:
                return dict(record)
            return None
            
        except Exception as e:
            logger.error(f"Error fetching LoRA details for '{lora_name}': {e}", exc_info=True)
            return None

async def sync_loras_to_db(api_loras: List[str]) -> Dict[str, int]:
    """
    Synchronize LoRA list from API to database.
    - Adds new LoRAs that don't exist in DB
    - Removes LoRAs from DB that no longer exist in API
    - Updates timestamp for existing LoRAs
    
    Returns dict with:
    - added: number of new LoRAs added
    - removed: number of old LoRAs removed
    - total: total LoRAs in database after sync
    """
    if not api_loras:
        logger.warning("sync_loras_to_db called with empty api_loras list")
        return {"added": 0, "removed": 0, "total": 0}
    
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        try:
            # Use sets for efficient comparison
            api_lora_set = set(api_loras)
            
            # Fetch current LoRAs from database
            records = await conn.fetch("SELECT name FROM loras;")
            db_lora_set = set(record['name'] for record in records)
            
            # Calculate differences
            loras_to_add = api_lora_set - db_lora_set
            loras_to_remove = db_lora_set - api_lora_set
            loras_to_update = api_lora_set & db_lora_set  # Existing ones
            
            # Start transaction
            async with conn.transaction():
                added_count = 0
                removed_count = 0
                updated_count = 0
                
                # Add new LoRAs
                if loras_to_add:
                    await conn.executemany(
                        """
                        INSERT INTO loras(name) 
                        VALUES($1) 
                        ON CONFLICT (name) DO NOTHING;
                        """,
                        [(name,) for name in loras_to_add]
                    )
                    added_count = len(loras_to_add)
                    logger.info(f"✓ Added {added_count} new LoRAs to database.")
                
                # Remove outdated LoRAs
                if loras_to_remove:
                    await conn.executemany(
                        "DELETE FROM loras WHERE name = $1;",
                        [(name,) for name in loras_to_remove]
                    )
                    removed_count = len(loras_to_remove)
                    logger.info(f"✓ Removed {removed_count} outdated LoRAs from database.")
                
                # Update timestamps for existing LoRAs
                if loras_to_update:
                    await conn.executemany(
                        """
                        UPDATE loras 
                        SET updated_at = NOW() 
                        WHERE name = $1;
                        """,
                        [(name,) for name in loras_to_update]
                    )
                    updated_count = len(loras_to_update)
                    logger.debug(f"Updated timestamps for {updated_count} existing LoRAs.")
            
            # Get final count
            total_count = await conn.fetchval("SELECT COUNT(*) FROM loras;")
            
            result = {
                "added": added_count,
                "removed": removed_count,
                "updated": updated_count,
                "total": total_count
            }
            
            logger.info(f"Sync complete: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error syncing LoRAs to database: {e}", exc_info=True)
            return {"added": 0, "removed": 0, "updated": 0, "total": 0}

async def clear_all_loras() -> int:
    """
    Clear all LoRAs from the database.
    Returns the number of rows deleted.
    WARNING: This is a destructive operation!
    """
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        try:
            result = await conn.execute("DELETE FROM loras;")
            # Extract row count from result string like "DELETE 42"
            count = int(result.split()[-1]) if result else 0
            logger.warning(f"⚠️ Cleared all LoRAs from database. Deleted: {count}")
            return count
            
        except Exception as e:
            logger.error(f"Error clearing LoRAs: {e}", exc_info=True)
            return 0

async def get_database_stats() -> Dict[str, Any]:
    """
    Get statistics about the database.
    Returns dict with various metrics.
    """
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        try:
            total_loras = await conn.fetchval("SELECT COUNT(*) FROM loras;")
            
            oldest = await conn.fetchrow("""
                SELECT name, added_at 
                FROM loras 
                ORDER BY added_at ASC 
                LIMIT 1;
            """)
            
            newest = await conn.fetchrow("""
                SELECT name, added_at 
                FROM loras 
                ORDER BY added_at DESC 
                LIMIT 1;
            """)
            
            stats = {
                "total_loras": total_loras,
                "oldest_lora": dict(oldest) if oldest else None,
                "newest_lora": dict(newest) if newest else None,
                "pool_size": pool.get_size(),
                "pool_max_size": pool.get_max_size(),
                "pool_min_size": pool.get_min_size()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}", exc_info=True)
            return {"error": str(e)}

# ===================================================================
# SEARCH & QUERY OPERATIONS
# ===================================================================
async def search_loras(query: str, limit: int = 20) -> List[str]:
    """
    Search for LoRAs by name (case-insensitive partial match).
    Returns list of matching LoRA names.
    """
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        try:
            records = await conn.fetch("""
                SELECT name 
                FROM loras 
                WHERE name ILIKE $1
                ORDER BY name ASC
                LIMIT $2;
            """, f"%{query}%", limit)
            
            return [record['name'] for record in records]
            
        except Exception as e:
            logger.error(f"Error searching LoRAs with query '{query}': {e}", exc_info=True)
            return []

async def lora_exists(lora_name: str) -> bool:
    """
    Check if a LoRA exists in the database.
    Returns True if exists, False otherwise.
    """
    pool = await get_pool()
    
    async with pool.acquire() as conn:
        try:
            exists = await conn.fetchval("""
                SELECT EXISTS(SELECT 1 FROM loras WHERE name = $1);
            """, lora_name)
            
            return bool(exists)
            
        except Exception as e:
            logger.error(f"Error checking if LoRA '{lora_name}' exists: {e}", exc_info=True)
            return False

# ===================================================================
# HEALTH CHECK
# ===================================================================
async def health_check() -> bool:
    """
    Check if database connection is healthy.
    Returns True if healthy, False otherwise.
    """
    try:
        pool = await get_pool()
        
        async with pool.acquire() as conn:
            # Simple query to test connection
            result = await conn.fetchval("SELECT 1;")
            return result == 1
            
    except Exception as e:
        logger.error(f"Database health check failed: {e}", exc_info=True)
        return False
