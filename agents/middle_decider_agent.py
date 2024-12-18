from autogen_core import RoutedAgent, rpc, MessageContext
from autogen_core.models import LLMMessage, SystemMessage, UserMessage, AssistantMessage
from autogen_core.tool_agent import tool_agent_caller_loop
from typing import List
from tools.function_tools import final_pipeline_tool


class MiddleDeciderAgent(RoutedAgent):
    def __init__(self, model_client):
        super().__init__(description="Middle Decider Agent")
        self.model_client = model_client

    @rpc
    async def decide_next_step(self, message: dict, ctx: MessageContext) -> dict:
        description_dict = message  # The small description dictionary

        input_messages: List[LLMMessage] = [
            SystemMessage(
                content="""Based on the provided data description, decide which processing function to call next.
Available function: final_pipeline."""
            ),
            UserMessage(content=str(description_dict), source="user"),
        ]

        tool_agent_id = await self.runtime.get("tool_agent_type", key="tool_agent")

        # Use the caller loop to decide the next pipeline
        generated_messages = await tool_agent_caller_loop(
            caller=self,
            tool_agent_id=tool_agent_id,
            model_client=self.model_client,
            input_messages=input_messages,
            tool_schema=[final_pipeline_tool.schema],
            cancellation_token=ctx.cancellation_token,
            caller_source="assistant",
        )

        # Extract decision info
        last_message_content = None
        for msg in reversed(generated_messages):
            if isinstance(msg, AssistantMessage):
                if isinstance(msg.content, str):
                    last_message_content = msg.content
                    break

        if last_message_content:
            import json

            decision_info = json.loads(last_message_content)
            return decision_info  # Returning decision info to the CoordinatorAgent

        return {}  # Default empty dict if unable to process
