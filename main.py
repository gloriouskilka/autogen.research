# import asyncio
# import random
# from dataclasses import dataclass
# from typing import List
# from loguru import logger
#
# from autogen_agentchat.teams._group_chat._sequential_routed_agent import (
#     SequentialRoutedAgent,
# )
# from autogen_core import (
#     AgentId,
#     DefaultTopicId,
#     MessageContext,
#     SingleThreadedAgentRuntime,
#     default_subscription,
#     message_handler,
# )
#
#
# @dataclass
# class Message:
#     content: str
#
#
# @default_subscription
# class _TestAgent(SequentialRoutedAgent):
#     def __init__(self, description: str) -> None:
#         super().__init__(description=description)
#         self.messages: List[Message] = []
#
#     @message_handler
#     async def handle_content_publish(
#         self, message: Message, ctx: MessageContext
#     ) -> None:
#         # Sleep a random amount of time to simulate processing time.
#         await asyncio.sleep(random.random() / 100)
#         self.messages.append(message)
#
#
# async def main() -> None:
#     runtime = SingleThreadedAgentRuntime()
#     runtime.start()
#     await _TestAgent.register(
#         runtime, type="test_agent", factory=lambda: _TestAgent(description="Test Agent")
#     )
#     test_agent_id = AgentId(type="test_agent", key="default")
#     for i in range(100):
#         await runtime.publish_message(
#             Message(content=f"{i}"), topic_id=DefaultTopicId()
#         )
#     await runtime.stop_when_idle()
#     test_agent = await runtime.try_get_underlying_agent_instance(
#         test_agent_id, _TestAgent
#     )
#     for i in range(100):
#         assert test_agent.messages[i].content == f"{i}"
#
#
# if __name__ == "__main__":
#     try:
#         asyncio.run(main())
#     except (KeyboardInterrupt, SystemExit):
#         logger.info("Bot stopped!")
import asyncio

from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()


# Define tools (implement execute_sql_query and analyze_data functions)
def execute_sql_query(query_name: str) -> dict:
    # Implementation to execute the SQL query and return results
    pass


def analyze_data(data: dict) -> dict:
    # Implementation to analyze data and return findings
    pass


# Define agents
planner_agent = AssistantAgent(
    name="PlannerAgent",
    model_client=OpenAIChatCompletionClient(
        model="gpt-4o", api_key=settings.openai_api_key
    ),
    handoffs=["SQLQueryAgent", "DataAnalysisAgent", "ExplanationAgent"],
    system_message="Coordinate the workflow to answer the user's question.",
)

sql_query_agent = AssistantAgent(
    name="SQLQueryAgent",
    model_client=OpenAIChatCompletionClient(
        model="gpt-4o", api_key=settings.openai_api_key
    ),
    tools=[execute_sql_query],
    handoffs=["PlannerAgent"],
    system_message="Execute predefined SQL queries to retrieve relevant data.",
)

data_analysis_agent = AssistantAgent(
    name="DataAnalysisAgent",
    model_client=OpenAIChatCompletionClient(
        model="gpt-4o", api_key=settings.openai_api_key
    ),
    tools=[analyze_data],
    handoffs=["PlannerAgent"],
    system_message="Analyze data to find insights.",
)

explanation_agent = AssistantAgent(
    name="ExplanationAgent",
    model_client=OpenAIChatCompletionClient(
        model="gpt-4o", api_key=settings.openai_api_key
    ),
    handoffs=[],
    system_message="Generate a clear explanation based on the analysis.",
)

# Set up the swarm
termination = TextMentionTermination("TERMINATE")
team = Swarm(
    participants=[
        planner_agent,
        sql_query_agent,
        data_analysis_agent,
        explanation_agent,
    ],
    termination_condition=termination,
)


async def run_swarm():
    # Запуск swarm
    await Console(
        team.run_stream(task="Why have users stopped buying large yellow hats?")
    )


# Запуск асинхронной функции
if __name__ == "__main__":
    asyncio.run(run_swarm())
