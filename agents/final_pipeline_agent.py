# final_pipeline_agent.py

from autogen_core import RoutedAgent, rpc, MessageContext
import pandas as pd


class FinalPipelineAgent(RoutedAgent):
    @rpc
    async def perform_final_analysis(self, message: dict, ctx: MessageContext) -> dict:
        dataframe_dict = message["dataframe"]
        df = pd.DataFrame.from_dict(dataframe_dict)
        # Perform analysis
        stats = df.describe().to_dict()
        final_overview = {"statistics": stats}
        return {"final_overview": final_overview}
