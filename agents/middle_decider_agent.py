from autogen_core import RoutedAgent, rpc, MessageContext, AgentId, message_handler
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import LLMMessage, SystemMessage, UserMessage, AssistantMessage
from autogen_core.tool_agent import tool_agent_caller_loop
from typing import List, Any

from agents.common import DescriptionDict, DecisionInfo

# from tools.function_tools import calculate_statistics_tool, final_pipeline_tool
import json


class MiddleDeciderAgent(RoutedAgent):
    def __init__(
        self,
        description: str,
        model_client,
        tool_agent_id: AgentId,
        tools: List[Any],
    ) -> None:
        super().__init__(description)
        self.model_client = model_client
        self.tool_agent_id = tool_agent_id
        self.tools = tools  # List of Tool or ToolSchema
        self.context = BufferedChatCompletionContext(buffer_size=10)

    @message_handler
    async def handle_description_dict(self, message: DescriptionDict, ctx: MessageContext) -> str:
        # Create a system message to instruct the assistant
        system_message = SystemMessage(content="Summarize the result of the previous steps.")
        # Add the user's input to the messages
        user_message = UserMessage(
            content=message.model_dump_json(),  # Convert input model to JSON string
            source=ctx.sender.key if ctx.sender else "user",
        )

        # Prepare the input messages for the model
        input_messages = [system_message, user_message]

        # Use the tool_agent_caller_loop to handle tool execution and LLM interaction
        generated_messages = await tool_agent_caller_loop(
            caller=self,
            tool_agent_id=self.tool_agent_id,
            model_client=self.model_client,
            input_messages=input_messages,
            tool_schema=self.tools,
            cancellation_token=ctx.cancellation_token,
            caller_source=self.id.key,  # Source identifier for messages
        )

        # Extract the assistant's final response
        assistant_response = [msg for msg in generated_messages if isinstance(msg, AssistantMessage)][-1]

        # Return the assistant's summary
        return assistant_response.content


# class MiddleDeciderAgent(RoutedAgent):
#     def __init__(self, model_client):
#         super().__init__(description="Middle Decider Agent")
#         self.model_client = model_client
#
#     @rpc
#     async def decide_next_step(self, message: DescriptionDict, ctx: MessageContext) -> DecisionInfo:
#         description_dict = message.description  # The small description dictionary
#
#         # Retrieve the ToolAgent ID
#         tool_agent_id = await self.runtime.get("tool_agent_type", key="default")
#
#         # Prepare the input messages for the LLM
#         input_messages: List[LLMMessage] = [
#             SystemMessage(
#                 content="""
# You are an assistant that decides which processing function to call next based on the provided data description.
# Available functions are:
# - calculate_statistics: Use this function to calculate statistical summaries from the data description.
# - final_pipeline: Use this function to perform final data processing.
# Your goal is to decide the best next step to process the data.
# """
#             ),
#             UserMessage(content=f"Data Description:\n{description_dict}", source="user"),
#         ]
#
#         # Use the tool_agent_caller_loop to interact with the LLM and execute function calls
#         generated_messages = await tool_agent_caller_loop(
#             caller=self,
#             tool_agent_id=tool_agent_id,
#             model_client=self.model_client,
#             input_messages=input_messages,
#             tool_schema=[calculate_statistics_tool.schema, final_pipeline_tool.schema],
#             cancellation_token=ctx.cancellation_token,
#             caller_source="assistant",
#         )
#
#         # Extract decision info from the last assistant message
#         last_message_content = None
#         for msg in reversed(generated_messages):
#             if isinstance(msg, AssistantMessage) and isinstance(msg.content, str):
#                 last_message_content = msg.content
#                 break
#
#         if last_message_content:
#             # Deserialize the result (assuming JSON format)
#             decision_info = json.loads(last_message_content)
#             return DecisionInfo(info=decision_info)  # Returning decision info to the CoordinatorAgent
#
#         return DecisionInfo(info={})  # Default empty dict if unable to process
