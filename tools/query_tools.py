from autogen_core.tools import FunctionTool
from utils.db_utils import execute_sql_query
from autogen_core import CancellationToken
from typing import Any


async def query_database(query_name: str, cancellation_token: CancellationToken = None) -> Any:
    sql_query = load_sql_query(query_name)
    results = execute_sql_query(sql_query)
    return results


def load_sql_query(query_name: str) -> str:
    query_file_path = f"./queries/{query_name}.sql"
    with open(query_file_path, "r", encoding="utf-8") as file:
        sql_query = file.read()
    return sql_query


query_tool = FunctionTool(func=query_database, description="Query the database using a specified SQL file name.")
