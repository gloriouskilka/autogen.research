import asyncio
from typing import Dict
from pydantic import BaseModel
from autogen_core import (
    RoutedAgent,
    message_handler,
    MessageContext,
    AgentId,
    SingleThreadedAgentRuntime,
    CancellationToken,
    AgentRuntime,
)

from utils.data_utils import process_data_pipeline_a, process_data_pipeline_b, analyze_full_data
from agents.common import PipelineResult, OverviewInfo


# Assume these are your processing functions
async def pipeline_a(data: str, cancellation_token: CancellationToken = None) -> PipelineResult:
    dataframe, description_dict = process_data_pipeline_a(data)
    return PipelineResult(dataframe=dataframe.to_dict(), description_dict=description_dict)


async def pipeline_b(data: str, cancellation_token: CancellationToken = None) -> PipelineResult:
    dataframe, description_dict = process_data_pipeline_b(data)
    return PipelineResult(dataframe=dataframe.to_dict(), description_dict=description_dict)


async def final_pipeline(dataframe: Dict, info: Dict, cancellation_token: CancellationToken = None) -> OverviewInfo:
    overview_info = analyze_full_data(dataframe)
    return overview_info


# tools/function_tools.py

from autogen_core.tools import FunctionTool
from autogen_core import CancellationToken

# from utils.data_utils import calculate_statistics


async def calculate_statistics(description_dict: dict, cancellation_token: CancellationToken = None) -> dict:
    # Implement your logic to calculate statistics from the description_dict
    # For example:
    # Add more statistical calculations as needed
    return {"num_columns": len(description_dict.get("columns", [])), "summary": description_dict.get("summary", "")}


calculate_statistics_tool = FunctionTool(
    func=calculate_statistics, description="Calculate statistical summaries from the data description."
)


# Define message types for your agent
class PipelineARequest(BaseModel):
    data: str


class PipelineBRequest(BaseModel):
    data: str


class FinalPipelineRequest(BaseModel):
    dataframe: Dict
    info: Dict


# Create the agent that handles these messages
class DataPipelineAgent(RoutedAgent):
    def __init__(self, description: str = "Data Pipeline Agent"):
        super().__init__(description)

    @message_handler
    async def handle_pipeline_a(self, message: PipelineARequest, ctx: MessageContext) -> PipelineResult:
        result = await pipeline_a(message.data, cancellation_token=ctx.cancellation_token)
        return result

    @message_handler
    async def handle_pipeline_b(self, message: PipelineBRequest, ctx: MessageContext) -> PipelineResult:
        result = await pipeline_b(message.data, cancellation_token=ctx.cancellation_token)
        return result

    @message_handler
    async def handle_final_pipeline(self, message: FinalPipelineRequest, ctx: MessageContext) -> OverviewInfo:
        result = await final_pipeline(message.dataframe, message.info, cancellation_token=ctx.cancellation_token)
        return result
