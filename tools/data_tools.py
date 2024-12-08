# data_tools.py

from autogen_core.tools import FunctionTool
from utils.db_utils import query_db
import pandas as pd


async def pipeline_a(data_id: int) -> dict:
    data = await query_db(data_id)
    # Perform analysis using pandas
    df = pd.DataFrame(data)
    overview = {"mean_values": df.mean().to_dict()}
    return {"dataframe": df.to_dict(), "overview": overview}


pipeline_a_tool = FunctionTool(func=pipeline_a, description="Process data using Pipeline A")
