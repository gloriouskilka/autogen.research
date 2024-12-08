import asyncio
import os
from typing import List, Dict, Any

from sqlalchemy import text

# +TODO: move to some other place

# DATABASE_URL = "sqlite+aiosqlite:///:memory:"
#
# # Create async engine
# engine = create_async_engine(DATABASE_URL, echo=False, future=True)
#
# # Create session factory
# AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
#
# # Import the text function
# from sqlalchemy.sql import text
#
#
# # Initialize the database
# async def init_db():
#     # Read SQL scripts from files
#     with open("books\\01_create.sql", "r") as f:
#         create_table_sql = f.read()
#     with open("books\\02_insert.sql", "r") as f:
#         insert_data_sql = f.read()
#
#     # Use the async engine to execute SQL statements
#     async with engine.begin() as conn:
#         # Split the create_table_sql into individual statements and execute them
#         for stmt in create_table_sql.strip().split(";"):
#             stmt = stmt.strip()
#             if stmt:
#                 await conn.execute(text(stmt))
#         # Split the insert_data_sql into individual statements and execute them
#         for stmt in insert_data_sql.strip().split(";"):
#             stmt = stmt.strip()
#             if stmt:
#                 await conn.execute(text(stmt))
#
#     # Optionally, fetch and display data to verify
#     async with AsyncSessionLocal() as session:
#         logger.debug("BOOKS Table:")
#         result = await session.execute(text("SELECT * FROM BOOKS"))
#         books_rows = result.fetchall()
#         for row in books_rows:
#             logger.debug(row)
#         logger.debug("\nINVENTORY Table:")
#         result = await session.execute(text("SELECT * FROM INVENTORY"))
#         inventory_rows = result.fetchall()
#         for row in inventory_rows:
#             logger.debug(row)
#
#
# -TODO: move to some other place


# Tool to execute SQL queries loaded from files
async def execute_sql_query(query_name: str) -> List[Dict[str, Any]]:
    """
    Execute an SQL query loaded from a file and return the results.
    """
    # Map of available query files
    query_files = {
        "excess_inventory": "03_query_01_excess.sql",
        "obsolete_inventory": "03_query_02_obsolete.sql",
        "top_selling_books": "03_query_03_top_selling.sql",
        "least_selling_books": "03_query_04_least_selling.sql",
        "inventory_turnover": "03_query_05_turnover.sql",
        "stock_aging": "03_query_06_stock_aging.sql",
        "forecast_demand": "03_query_07_forecast_demand.sql",
        "supplier_performance": "03_query_08_supplier_performance.sql",
        "seasonal_trends": "03_query_09_seasonal_trends.sql",
        "profit_margins": "03_query_10_profit_margins.sql",
        "warehouse_optimization": "03_query_11_warehouse_space_optimize.sql",
        "return_rates": "03_query_12_return_reasons.sql",
    }

    if query_name not in query_files:
        raise ValueError(f"Query '{query_name}' not found. Available queries: {list(query_files.keys())}")

    query_file = query_files[query_name]

    # Read the SQL query from the file
    with open("database\\queries\\" + query_file, "r") as f:
        query_sql = f.read()

    # Execute the SQL query
    async with AsyncSessionLocal() as session:  # TODO: NOW: Need to use AsyncSessionLocal, which should be defined in some other place
        result = await session.execute(text(query_sql))
        rows = result.fetchall()

        # Get column names from the result cursor
        column_names = result.keys()

        # Convert rows to list of dictionaries
        data = [dict(zip(column_names, row)) for row in rows]

    return data


# async def execute_sql_query(query_name: str) -> List[Dict[str, Any]]:
#     """
#     Execute an SQL query loaded from a file and return the results.
#     """
#     query_files = {
#         "excess_inventory": "03_query_01_excess.sql",
#         "obsolete_inventory": "03_query_02_obsolete.sql",
#         "top_selling_books": "03_query_03_top_selling.sql",
#         "least_selling_books": "03_query_04_least_selling.sql",
#         "inventory_turnover": "03_query_05_turnover.sql",
#         "stock_aging": "03_query_06_stock_aging.sql",
#         "forecast_demand": "03_query_07_forecast_demand.sql",
#         "supplier_performance": "03_query_08_supplier_performance.sql",
#         "seasonal_trends": "03_query_09_seasonal_trends.sql",
#         "profit_margins": "03_query_10_profit_margins.sql",
#         "warehouse_optimization": "03_query_11_warehouse_space_optimize.sql",
#         "return_rates": "03_query_12_return_reasons.sql",
#     }
#
#     if query_name not in query_files:
#         raise ValueError(f"Query '{query_name}' not found. Available queries: {list(query_files.keys())}")
#
#     query_file = query_files[query_name]
#
#     # Read the SQL query from the file
#     with open(query_file, "r") as f:
#         query_sql = f.read()
#
#     # Execute the SQL query asynchronously
#     # Use your preferred async database library (e.g., aiosqlite, asyncpg)
#     # For example, using aiosqlite:
#     import aiosqlite
#
#     async with aiosqlite.connect("your_database.db") as db:
#         cursor = await db.execute(query_sql)
#         rows = await cursor.fetchall()
#         # Get column names
#         column_names = [description[0] for description in cursor.description]
#         results = [dict(zip(column_names, row)) for row in rows]
#         await cursor.close()
#
#     return results
