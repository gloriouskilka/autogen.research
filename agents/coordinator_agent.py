# agents/coordinator_agent.py
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import RoutedAgent, rpc, MessageContext, AgentId
from autogen_core.models import LLMMessage, SystemMessage, UserMessage, AssistantMessage
from autogen_core.tool_agent import tool_agent_caller_loop
from loguru import logger

from agents.common import UserInput, FinalResult, PipelineResult, DecisionInfo, FinalPipelineInput, DescriptionDict
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

    def verify_pipeline_a_input_correctness(self, sock_type: str) -> bool:
        if sock_type not in ["wool", "cotton", "polyester"]:
            return False
        return HandoffMessage(target="pipeline_a", content=sock_type)

    def verify_pipeline_b_input_correctness(self, list_of_apple_juices: List[str]) -> bool:
        if any("apple" not in juice for juice in list_of_apple_juices):
            return False
        return HandoffMessage(target="pipeline_b", content=str(list_of_apple_juices))

    @rpc
    async def handle_user_input(self, message: UserInput, ctx: MessageContext) -> FinalResult:
        pipeline_selector = AssistantAgent(
            name="pipeline_selector",
            model_client=self.model_client,
            system_message=f"Choose a pipeline to run: pipeline_a or pipeline_b. Always verify the input data before proceeding. If both pipelines are valid, pipeline_a will be selected. If neither pipeline is valid, an error will be returned.",
            tools=[self.verify_pipeline_a_input_correctness, self.verify_pipeline_b_input_correctness],
            handoffs=["pipeline_a", "pipeline_b", "error_selecting"],  # TODO: error_selecting doesn't work
        )
        result = await pipeline_selector.run(task=message.text)
        content = result.messages[-1].content

        agent_id = AgentId(type="data_pipeline_agent", key="default")

        if content == "pipeline_a":
            request_a = PipelineARequest(data="some input data")
            pipeline_result = await self.send_message(message=request_a, recipient=agent_id)
            logger.debug(f"Pipeline A result: {pipeline_result}")
        elif content == "pipeline_b":
            request_b = PipelineBRequest(data="some other input data")
            pipeline_result = await self.send_message(message=request_b, recipient=agent_id)
            logger.debug(f"Pipeline B result: {pipeline_result}")
        else:
            return FinalResult(result="Error: Invalid pipeline selection.")

        middle_decider_agent_id = await self.runtime.get("middle_decider_agent_type", key="default")
        decision_info = await self.send_message(
            message=DescriptionDict(description=pipeline_result.description_dict),
            recipient=middle_decider_agent_id,
            cancellation_token=ctx.cancellation_token,
        )

        # Proceed to final pipeline
        final_pipeline_agent_id = await self.runtime.get("final_pipeline_agent_type", key="default")
        final_input = FinalPipelineInput(dataframe=pipeline_result.dataframe, info=decision_info.info)

        final_result = await self.send_message(
            message=final_input, recipient=final_pipeline_agent_id, cancellation_token=ctx.cancellation_token
        )

        return FinalResult(result=final_result.result)
