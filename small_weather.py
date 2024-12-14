from loguru import logger
import sys

# Define a tool that gets the weather for a city.
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from util import model_client

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
    level="INFO",
)


async def get_weather(city: str) -> str:
    """Get the weather for a city."""
    weather_info = f"The weather in {city} is 72 degrees and Sunny."
    logger.info(f"Retrieved weather for {city}: {weather_info}")
    return weather_info


# Create an assistant agent.
weather_agent = AssistantAgent(
    "assistant",
    model_client=model_client,
    tools=[get_weather],
    system_message="Respond 'TERMINATE' when task is complete.",
)
logger.info("Assistant agent created.")

# Define a termination condition.
text_termination = TextMentionTermination("TERMINATE")
logger.info("Termination condition defined.")

# Create a single-agent team.
single_agent_team = RoundRobinGroupChat([weather_agent], termination_condition=text_termination)
logger.info("Single-agent team created.")


async def run_team() -> None:
    logger.info("Starting team run.")
    result = await single_agent_team.run(task="What is the weather in New York?")
    logger.info(f"Team run completed with result: {result}")


# If you're running this script directly, you can use asyncio to run the async function
if __name__ == "__main__":
    import asyncio

    asyncio.run(run_team())
