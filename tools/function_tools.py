from autogen_core.tools import FunctionTool
from autogen_core import CancellationToken
from typing import Dict, Any
from utils.db_utils import execute_sql_query
from utils.data_utils import (
    process_data_pipeline_a,
    process_data_pipeline_b,
    analyze_full_data,
)
import pandas as pd


async def pipeline_a(data: pd.DataFrame, cancellation_token: CancellationToken = None) -> Dict[str, Any]:
    dataframe, description_dict = process_data_pipeline_a(data)
    return {"dataframe": dataframe.to_dict(), "description_dict": description_dict}


async def pipeline_b(data: pd.DataFrame, cancellation_token: CancellationToken = None) -> Dict[str, Any]:
    dataframe, description_dict = process_data_pipeline_b(data)
    return {"dataframe": dataframe.to_dict(), "description_dict": description_dict}


async def final_pipeline(dataframe: pd.DataFrame, cancellation_token: CancellationToken = None) -> Dict[str, Any]:
    overview_info = analyze_full_data(dataframe)
    return {"overview_info": overview_info}


# Wrap functions as tools
pipeline_a_tool = FunctionTool(func=pipeline_a, description="Process data using Pipeline A.")
pipeline_b_tool = FunctionTool(func=pipeline_b, description="Process data using Pipeline B.")
final_pipeline_tool = FunctionTool(func=final_pipeline, description="Execute the final analysis pipeline.")
