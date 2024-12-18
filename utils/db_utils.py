# db_utils.py

import asyncpg


async def query_db(data_id: int) -> list:
    conn = await asyncpg.connect(database="your_db")
    data = await conn.fetch("SELECT * FROM your_table WHERE id = $1", data_id)
    await conn.close()
    return data
