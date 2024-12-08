import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Handoff
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from loguru import logger

from util import model_client

# Create a lazy assistant agent that always hands off to the user.
lazy_agent = AssistantAgent(
    "lazy_assistant",
    model_client=model_client,
    handoffs=[Handoff(target="user", message="Transfer to user.")],
    system_message="Always transfer to user when you don't know the answer. Respond 'TERMINATE' when task is complete.",
)

# Define a termination condition that checks for handoff message targetting helper and text "TERMINATE".
handoff_termination = HandoffTermination(target="user")
text_termination = TextMentionTermination("TERMINATE")
termination = handoff_termination | text_termination

# Create a single-agent team.
lazy_agent_team = RoundRobinGroupChat([lazy_agent], termination_condition=termination)


async def main() -> None:
    # for m in lazy_agent_team.run_stream(task="It is raining in New York."):
    async for m in lazy_agent_team.run_stream(task="It is raining in New York."):
        i = 100
    #     Print m


# Use `asyncio.run(Console(lazy_agent_team.run_stream(task="What is the weather in New York?")))` when running in a script.
# await Console(lazy_agent_team.run_stream(task="What is the weather in New York?"))
# asyncio.run(Console(lazy_agent_team.run_stream(task="It is raining in New York.")))
# await Console(lazy_agent_team.run_stream(task="It is raining in New York."))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped!")
