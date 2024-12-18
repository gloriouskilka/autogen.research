from autogen_core import RoutedAgent, rpc, MessageContext
from autogen_core.models import LLMMessage, SystemMessage, UserMessage, AssistantMessage
from autogen_core.tool_agent import tool_agent_caller_loop
from agents.common import UserInput, FinalResult
from typing import List
import json


class CoordinatorAgent(RoutedAgent):
    def __init__(self, model_client):
        super().__init__(description="Coordinator Agent")
        self.model_client = model_client

    @rpc
    async def handle_user_input(self, message: UserInput, ctx: MessageContext) -> FinalResult:
        user_text = message.text

        input_messages: List[LLMMessage] = [
            SystemMessage(
                content="""You are an assistant that decides which SQL query to execute based on user input.
                Available queries are query_01_excess, query_02_obsolete, query_03_top_selling, etc."""
            ),
            UserMessage(content=user_text, source="user"),
        ]

        tool_agent_id = await self.runtime.get("tool_agent_type", key="tool_agent")

        # Use the caller loop to decide which query to run
        generated_messages = await tool_agent_caller_loop(
            caller=self,
            tool_agent_id=tool_agent_id,
            model_client=self.model_client,
            input_messages=input_messages,
            tool_schema=[query_tool.schema],
            cancellation_token=ctx.cancellation_token,
            caller_source="assistant",
        )

        # Extract the query result
        last_message = generated_messages[-1]
        if isinstance(last_message, AssistantMessage) and isinstance(last_message.content, str):
            try:
                result_data = json.loads(last_message.content)
                dataframe_dict = result_data
                # Proceed to the LLM-decided pipeline
                middle_decider_agent_id = await self.runtime.get(
                    "middle_decider_agent_type", key="middle_decider_agent"
                )
                decision_info = await self.send_message(
                    message={"dataframe": dataframe_dict},
                    recipient=middle_decider_agent_id,
                    cancellation_token=ctx.cancellation_token,
                )

                return decision_info  # FinalResult will be returned from the final pipeline
            except json.JSONDecodeError:
                return FinalResult(result="Error: Failed to parse the query result.")
        else:
            return FinalResult(result="Error: Unable to process input.")
