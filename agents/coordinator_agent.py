# agents/coordinator_agent.py

from autogen_core import RoutedAgent, rpc, MessageContext, AgentId
from autogen_core.models import LLMMessage, SystemMessage, UserMessage, AssistantMessage
from autogen_core.tool_agent import tool_agent_caller_loop
from loguru import logger

from agents.common import UserInput, FinalResult, PipelineResult, DecisionInfo, FinalPipelineInput
from typing import List
from tools.function_tools import (
    # pipeline_a_tool,
    # pipeline_b_tool,
    PipelineARequest,
    PipelineBRequest,
    FinalPipelineRequest,
)
import json


class CoordinatorAgent(RoutedAgent):
    def __init__(self, model_client):
        super().__init__(description="Coordinator Agent")
        self.model_client = model_client

    @rpc
    async def handle_user_input(self, message: UserInput, ctx: MessageContext) -> FinalResult:
        user_text = message.text

        #         input_messages: List[LLMMessage] = [
        #             SystemMessage(
        #                 content="""
        # You are an assistant that decides which initial processing function to call based on user input.
        # Available functions are pipeline_a and pipeline_b.
        # """
        #             ),
        #             UserMessage(content=user_text, source="user"),
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

        # # # tool_agent_id = await self.runtime.get("tool_agent_type", key="tool_agent")
        # tool_agent_id = await self.runtime.get("tool_agent_type", key="default")
        #
        # # Use the caller loop to decide initial pipeline
        # generated_messages = await tool_agent_caller_loop(
        #     caller=self,
        #     tool_agent_id=tool_agent_id,
        #     model_client=self.model_client,
        #     input_messages=input_messages,
        #     tool_schema=[pipeline_a_tool.schema, pipeline_b_tool.schema],
        #     cancellation_token=ctx.cancellation_token,
        #     caller_source="assistant",
        # )

        # Extract result data
        last_message_content = None
        for msg in reversed(generated_messages):
            if isinstance(msg, AssistantMessage):
                if isinstance(msg.content, str):
                    last_message_content = msg.content
                    break
                elif isinstance(msg.content, list):
                    continue  # Skip function calls

        if last_message_content:
            # Deserialize the result (assuming JSON format)
            result_data = json.loads(last_message_content)
            pipeline_result = PipelineResult(
                dataframe=result_data["dataframe"], description_dict=result_data["description_dict"]
            )

            # Proceed to the middle decider agent
            # middle_decider_agent_id = await self.runtime.get("middle_decider_agent_type", key="middle_decider_agent")
            middle_decider_agent_id = await self.runtime.get("middle_decider_agent_type")
            decision_info = await self.send_message(
                message=pipeline_result.description_dict,
                recipient=middle_decider_agent_id,
                cancellation_token=ctx.cancellation_token,
            )

            # Proceed to final pipeline
            # final_pipeline_agent_id = await self.runtime.get("final_pipeline_agent_type", key="final_pipeline_agent")
            final_pipeline_agent_id = await self.runtime.get("final_pipeline_agent_type")
            final_input = FinalPipelineInput(dataframe=pipeline_result.dataframe, info=decision_info.info)

            final_result = await self.send_message(
                message=final_input, recipient=final_pipeline_agent_id, cancellation_token=ctx.cancellation_token
            )

            return FinalResult(result=final_result.result)

        return FinalResult(result="Error: Unable to process input.")
