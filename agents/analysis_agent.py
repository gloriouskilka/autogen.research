# analysis_agent.py

from autogen_core import RoutedAgent, rpc, MessageContext
from autogen_core.models import SystemMessage, UserMessage


class AnalysisAgent(RoutedAgent):
    def __init__(self, model_client):
        super().__init__(description="Analysis Agent")
        self.model_client = model_client

    @rpc
    async def generate_report(self, message: dict, ctx: MessageContext) -> dict:
        final_overview = message["final_overview"]
        input_messages = [
            SystemMessage(content="Generate a report based on the final overview."),
            UserMessage(content=str(final_overview)),
        ]
        response = await self.model_client.create(messages=input_messages, cancellation_token=ctx.cancellation_token)
        report = response.content if isinstance(response.content, str) else "Error in generating report."
        return {"report": report}
