# agents/coordinator_agent.py
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Handoff
from autogen_agentchat.conditions import MaxMessageTermination, HandoffTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import RoundRobinGroupChat, Swarm
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

from autogen_core import AgentId, CancellationToken
from autogen_core import DefaultTopicId, MessageContext

# from autogen_core.agent import RoutedAgent
from autogen_agentchat.agents import AssistantAgent, Handoff
from autogen_agentchat.base import Response, TaskResult
from autogen_agentchat.messages import HandoffMessage, TextMessage
import logging


class CoordinatorAgent(RoutedAgent):
    def __init__(self, model_client):
        super().__init__(description="Coordinator Agent")
        self.model_client = model_client

    # Define your verification functions as tools
    async def verify_pipeline_a_input_correctness(self, sock_type: str) -> str | bool:
        if not any(it in sock_type for it in ["wool", "cotton", "polyester"]):
            return False
        return HandoffMessage(content=sock_type, source="assistant", target="pipeline_a")

    async def verify_pipeline_b_input_correctness(self, list_of_apple_juices: List[str]) -> List[str] | bool:
        if any("apple" not in juice for juice in list_of_apple_juices):
            return False
        return HandoffMessage(content=str(list_of_apple_juices), source="assistant", target="pipeline_a")

    @rpc
    async def handle_user_input(self, message: UserInput, ctx: MessageContext) -> FinalResult:
        # Define the assistant agent
        pipeline_selector = AssistantAgent(
            name="pipeline_selector",
            model_client=self.model_client,
            system_message=(
                "You are a coordinator assistant that decides which pipeline to use based on the user's input.\n"
                "Use the available tools to verify the input data before proceeding.\n"
                "If both pipelines are valid, select pipeline_a.\n"
                "If neither pipeline is valid, hand off to the user with an error message.\n"
                "Include the parsed parameters in the handoff message using {parameters}."
            ),
            tools=[self.verify_pipeline_a_input_correctness, self.verify_pipeline_b_input_correctness],
            handoffs=[
                Handoff(target="pipeline_a", message="pipeline_a"),
                Handoff(target="pipeline_b", message="pipeline_b"),
                Handoff(target="user", message="user"),
            ],
        )

        # Set up termination conditions for the assistant agent
        termination = (
            HandoffTermination(target="pipeline_a")
            | HandoffTermination(target="pipeline_b")
            | HandoffTermination(target="user")
            | MaxMessageTermination(5)
        )

        # Create a Swarm with the assistant agent
        team = Swarm(participants=[pipeline_selector], termination_condition=termination)

        # Run the Swarm with the user's task
        result = await team.run(task=message.text)

        # Get the last message from the assistant agent
        last_message = result.messages[-1]

        # Check if the last message is a HandoffMessage
        if isinstance(last_message, HandoffMessage):
            handoff_msg = last_message
            if handoff_msg.target == "pipeline_a":
                # Process handoff to pipeline_a
                data = handoff_msg.content  # Extract parameters if needed
                pipeline_result = await self.send_message(
                    message=PipelineARequest(data=data),
                    recipient=AgentId(type="data_pipeline_agent", key="default"),
                    cancellation_token=ctx.cancellation_token,
                )
                logger.debug(f"Pipeline A result: {pipeline_result}")
            elif handoff_msg.target == "pipeline_b":
                # Process handoff to pipeline_b
                data = handoff_msg.content
                pipeline_result = await self.send_message(
                    message=PipelineBRequest(data=data),
                    recipient=AgentId(type="data_pipeline_agent", key="default"),
                    cancellation_token=ctx.cancellation_token,
                )
                logger.debug(f"Pipeline B result: {pipeline_result}")
            elif handoff_msg.target == "user":
                # Unable to decide, hand off to user
                return FinalResult(result="Error: Unable to decide. Please provide more information.")
            else:
                return FinalResult(result="Error: Invalid target in handoff message.")
        else:
            # Handle unexpected message types
            return FinalResult(result=f"Unexpected response: {last_message.content}")

        # Proceed to the middle decider agent
        middle_decider_agent_id = await self.runtime.get("middle_decider_agent_type", key="default")
        decision_info: str = await self.send_message(
            message=DescriptionDict(description=pipeline_result.description_dict),
            recipient=middle_decider_agent_id,
            cancellation_token=ctx.cancellation_token,
        )

        # Proceed to final pipeline
        final_pipeline_agent_id = await self.runtime.get("final_pipeline_agent_type", key="default")
        final_input = FinalPipelineInput(dataframe=pipeline_result.dataframe, info={"decision": decision_info})

        final_result = await self.send_message(
            message=final_input,
            recipient=final_pipeline_agent_id,
            cancellation_token=ctx.cancellation_token,
        )

        return FinalResult(result=final_result.result)
