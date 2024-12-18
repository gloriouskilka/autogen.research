from autogen_core import RoutedAgent, rpc, MessageContext
from agents.common import FinalResult
from utils.data_utils import analyze_full_data
from typing import Dict
import pandas as pd


class FinalPipelineAgent(RoutedAgent):
    def __init__(self):
        super().__init__(description="Final Pipeline Agent")

    @rpc
    async def run_final_pipeline(self, message: Dict, ctx: MessageContext) -> FinalResult:
        dataframe_dict = message["dataframe"]
        info = message.get("info", {})
        dataframe = pd.DataFrame.from_dict(dataframe_dict)

        # Perform final deterministic processing
        overview_info = analyze_full_data(dataframe_dict)

        # Proceed to the analysis agent
        analysis_agent_id = await self.runtime.get("analysis_agent_type", key="analysis_agent")
        analysis_result = await self.send_message(
            message=overview_info, recipient=analysis_agent_id, cancellation_token=ctx.cancellation_token
        )

        return FinalResult(result=analysis_result)
