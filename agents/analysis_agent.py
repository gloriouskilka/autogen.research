from autogen_core import RoutedAgent, rpc, MessageContext
from autogen_core.models import LLMMessage, SystemMessage, UserMessage
from agents.common import FinalResult
from typing import List


class AnalysisAgent(RoutedAgent):
    def __init__(self, model_client):
        super().__init__(description="Analysis Agent")
        self.model_client = model_client

    @rpc
    async def generate_report(self, message: dict, ctx: MessageContext) -> str:
        overview_info = message  # The small overview dict

        input_messages: List[LLMMessage] = [
            SystemMessage(
                content="""You are a data analyst. Based on the following data analysis overview, provide actionable recommendations and expected results in natural language."""
            ),
            UserMessage(content=str(overview_info), source="user"),
        ]

        response = await self.model_client.create(
            messages=input_messages,
            cancellation_token=ctx.cancellation_token,
        )

        if isinstance(response.content, str):
            return response.content

        return "Error: Failed to generate report."
