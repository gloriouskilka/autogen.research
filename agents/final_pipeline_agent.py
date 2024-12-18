from autogen_core import RoutedAgent, rpc, MessageContext
from agents.common import FinalResult
import pandas as pd
from utils.data_utils import analyze_full_data


class FinalPipelineAgent(RoutedAgent):
    def __init__(self):
        super().__init__(description="Final Pipeline Agent")

    @rpc
    async def run_final_pipeline(self, message: dict, ctx: MessageContext) -> FinalResult:
        dataframe_dict = message["dataframe"]
        dataframe = pd.DataFrame.from_dict(dataframe_dict)
        description = message.get("description", {})

        # Run the final deterministic analysis
        overview_info = analyze_full_data(dataframe)

        # Proceed to the analysis agent
        analysis_agent_id = await self.runtime.get("analysis_agent_type", key="analysis_agent")
        analysis_result = await self.send_message(
            message=overview_info,
            recipient=analysis_agent_id,
            cancellation_token=ctx.cancellation_token,
        )

        return FinalResult(result=analysis_result)
