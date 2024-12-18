from autogen_core import RoutedAgent, rpc, MessageContext, AgentId
from autogen_core.models import LLMMessage, SystemMessage, UserMessage, AssistantMessage
from autogen_core.tool_agent import tool_agent_caller_loop
from typing import List

from loguru import logger

from tools.function_tools import PipelineARequest, PipelineBRequest, FinalPipelineRequest
import json

from agents.common import DescriptionDict, DecisionInfo


class MiddleDeciderAgent(RoutedAgent):
    def __init__(self, model_client):
        super().__init__(description="Middle Decider Agent")
        self.model_client = model_client

    @rpc
    async def decide_next_step(self, message: DescriptionDict, ctx: MessageContext) -> DecisionInfo:
        description_dict = message.description  # The small description dictionary

        #         input_messages: List[LLMMessage] = [
        #             SystemMessage(
        #                 content="""Based on the provided data description, decide which processing function to call next.
        # Available function: final_pipeline."""
        #             ),
        #             UserMessage(content=str(description_dict), source="user"),
        #         ]

        agent_id = AgentId(type="data_pipeline_agent", key="default")

        # Send messages to the agent
        request_a = PipelineARequest(data="some input data")
        result_a = await self.send_message(message=request_a, recipient=agent_id)
        logger.debug(f"Pipeline A result: {result_a}")

        request_b = PipelineBRequest(data="some other input data")
        result_b = await self.send_message(message=request_b, recipient=agent_id)
        logger.debug(f"Pipeline B result: {result_b}")

        # For the final pipeline
        request_final = FinalPipelineRequest(
            dataframe=result_a.dataframe, info=result_a.description_dict  # Assuming result_a is of type PipelineResult
        )
        result_final = await self.send_message(message=request_final, recipient=agent_id)

        # # tool_agent_id = await self.runtime.get("tool_agent_type", key="tool_agent")
        # tool_agent_id = await self.runtime.get("data_pipeline_agent", key="default")
        #
        # # Use the caller loop to decide the next pipeline
        # generated_messages = await tool_agent_caller_loop(
        #     caller=self,
        #     tool_agent_id=tool_agent_id,
        #     model_client=self.model_client,
        #     input_messages=input_messages,
        #     tool_schema=[final_pipeline_tool.schema],
        #     cancellation_token=ctx.cancellation_token,
        #     caller_source="assistant",
        # )

        # Extract decision info
        last_message_content = None
        for msg in reversed(generated_messages):
            if isinstance(msg, AssistantMessage):
                if isinstance(msg.content, str):
                    last_message_content = msg.content
                    break

        if last_message_content:
            decision_info = json.loads(last_message_content)
            return DecisionInfo(info=decision_info)  # Returning decision info to the CoordinatorAgent

        return DecisionInfo(info={})  # Default empty dict if unable to process
