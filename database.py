# Filename: database.py
import os
import asyncpg
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

DATABASE_URL = "postgres://mouse:sP7=hH8+rK3=gL4=pJ8_@/nursing-plum-pony"
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set!")

async def init_db():
    """Membuat tabel 'loras' jika belum ada."""
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS loras (
                name TEXT PRIMARY KEY,
                description TEXT,
                added_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        logger.info("Database initialized successfully. 'loras' table is ready.")
    finally:
        await conn.close()

async def get_loras_from_db() -> List[str]:
    """Mengambil semua nama LoRA dari database."""
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        records = await conn.fetch("SELECT name FROM loras ORDER BY name;")
        return [record['name'] for record in records]
    finally:
        await conn.close()

async def sync_loras_to_db(api_loras: List[str]) -> Dict[str, int]:
    """
    Menyinkronkan daftar LoRA dari API ke database.
    - Menambahkan LoRA baru.
    - Menghapus LoRA yang sudah tidak ada di API.
    """
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        # Menggunakan 'set' untuk operasi perbandingan yang efisien
        api_lora_set = set(api_loras)
        db_lora_set = set(await get_loras_from_db())

        loras_to_add = api_lora_set - db_lora_set
        loras_to_remove = db_lora_set - api_lora_set

        async with conn.transaction():
            # Menambahkan LoRA baru secara massal
            if loras_to_add:
                await conn.executemany(
                    "INSERT INTO loras(name) VALUES($1) ON CONFLICT (name) DO NOTHING;",
                    [(name,) for name in loras_to_add]
                )
                logger.info(f"Added {len(loras_to_add)} new LoRAs to the database.")

            # Menghapus LoRA lama secara massal
            if loras_to_remove:
                await conn.executemany(
                    "DELETE FROM loras WHERE name = $1;",
                    [(name,) for name in loras_to_remove]
                )
                logger.info(f"Removed {len(loras_to_remove)} outdated LoRAs from the database.")
        
        return {"added": len(loras_to_add), "removed": len(loras_to_remove)}

    finally:
        await conn.close()
