import sqlite3
import pandas as pd
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from utils.settings import settings


# Create async engine
engine = create_async_engine(settings.database_url, echo=False, future=True)

# Create session factory
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Import the text function
from sqlalchemy.sql import text


async def execute_sql_query(sql_query: str) -> pd.DataFrame:
    async with engine.begin() as conn:
        result = await conn.execute(text(sql_query))
        df = pd.DataFrame(await result.fetchall())
        df.columns = result.keys()

    return df
