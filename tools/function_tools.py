# tools/function_tools.py

from autogen_core.tools import FunctionTool
from typing import Dict, Any
from utils.data_utils import (
    process_data_pipeline_a,
    process_data_pipeline_b,
    analyze_full_data,
)
from autogen_core import CancellationToken


async def pipeline_a(data: str, cancellation_token: CancellationToken = None) -> Dict[str, Any]:
    dataframe, description_dict = process_data_pipeline_a(data)
    return {"dataframe": dataframe.to_dict(), "description_dict": description_dict}


async def pipeline_b(data: str, cancellation_token: CancellationToken = None) -> Dict[str, Any]:
    dataframe, description_dict = process_data_pipeline_b(data)
    return {"dataframe": dataframe.to_dict(), "description_dict": description_dict}


async def final_pipeline(dataframe: Dict, info: Dict, cancellation_token: CancellationToken = None) -> Dict[str, Any]:
    overview_info = analyze_full_data(dataframe)
    return {"overview_info": overview_info}


# Wrap functions as tools
pipeline_a_tool = FunctionTool(func=pipeline_a, description="Process data using Pipeline A.")
pipeline_b_tool = FunctionTool(func=pipeline_b, description="Process data using Pipeline B.")
final_pipeline_tool = FunctionTool(func=final_pipeline, description="Execute the final data processing pipeline.")
