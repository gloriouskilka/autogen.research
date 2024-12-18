from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import RoutedAgent, rpc, MessageContext, AgentId, message_handler, default_subscription
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import LLMMessage, SystemMessage, UserMessage, AssistantMessage, ChatCompletionClient
from autogen_core.tool_agent import tool_agent_caller_loop
from typing import List, Any, Callable, Awaitable

from autogen_core.tools import Tool
from loguru import logger

from agents.common import DescriptionDict, DecisionInfo

import json


class MiddleDeciderAgent(RoutedAgent):
    def __init__(
        self,
        description: str,
        model_client: ChatCompletionClient,
        tools: List[Tool | Callable[..., Any] | Callable[..., Awaitable[Any]]] | None = None,
    ) -> None:
        super().__init__(description)
        self.model_client = model_client
        self.tools = tools  # List of Tool or ToolSchema
        self.context = BufferedChatCompletionContext(buffer_size=10)

    @message_handler
    async def handle_description_dict(self, message: DescriptionDict, ctx: MessageContext) -> str:
        summary_agent = AssistantAgent(
            name="summary_llm_agent",
            model_client=self.model_client,
            tools=self.tools,
            system_message="Summarize the result of the previous steps. Write what the user can do next.",
        )
        team = RoundRobinGroupChat([summary_agent], termination_condition=MaxMessageTermination(3))

        result: TaskResult = await team.run(task="Count from 1 to 10, respond one at a time.")
        logger.debug(result)

        content = result.messages[-1].content
        return content
