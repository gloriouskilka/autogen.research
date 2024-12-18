# coordinator_agent.py

from autogen_core import RoutedAgent, rpc, MessageContext
from autogen_core.models import SystemMessage, UserMessage
from autogen_core.tool_agent import tool_agent_caller_loop
from tools.data_tools import pipeline_a_tool, pipeline_b_tool


class CoordinatorAgent(RoutedAgent):
    def __init__(self, model_client):
        super().__init__(description="Coordinator Agent")
        self.model_client = model_client

    @rpc
    async def handle_user_input(self, message: dict, ctx: MessageContext) -> dict:
        user_text = message["text"]
        input_messages = [
            SystemMessage(content="Decide which pipeline to use based on the user input."),
            UserMessage(content=user_text),
        ]
        tool_agent_id = await self.runtime.get("tool_agent", key="default")

        generated_messages = await tool_agent_caller_loop(
            caller=self,
            tool_agent_id=tool_agent_id,
            model_client=self.model_client,
            input_messages=input_messages,
            tool_schema=[pipeline_a_tool.schema, pipeline_b_tool.schema],
            cancellation_token=ctx.cancellation_token,
            caller_source=self.name,
        )
        # Process the LLM's decision and proceed accordingly
        # ...
        return {"status": "Pipeline selected and executed"}
