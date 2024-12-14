from loguru import logger
import sys

# Define a tool that gets the weather for a city.
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from util import model_client

import aiohttp


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


# Prompt used:
# Write a detailed Function description (it should be a Python code with Python documentation abilities used) explaining parameters in detail, write several examples of parameter values, returned values. This description will be used in a very sensitive context, the logic of a function will be regenerated based on description via LLM, so the function description should be very detailed, not to allow LLM to hallucinate.


async def get_weather(city: str) -> str:
    """
    Asynchronously fetches the current weather information for a specified city.

    Parameters:
        city (str):
            - Description: The name of the city to retrieve weather data for.
            - Requirements:
                * Must be a non-empty string.
                * Should correspond to a valid city recognized by the weather API.
            - Example Values:
                - "New York"
                - "Paris"
                - "Tokyo"
                - "São Paulo"

    Returns:
        str:
            - Description: A formatted string containing the current temperature and weather condition of the specified city.
            - Format: "The weather in {city} is {temperature} degrees and {condition}."
            - Example Returns:
                - "The weather in New York is 22 degrees and sunny."
                - "The weather in Paris is 18 degrees and cloudy."

    Raises:
        ValueError:
            - Condition: If the 'city' parameter is an empty string.
            - Message: "City name must be a non-empty string."
        aiohttp.ClientError:
            - Condition: If a network-related error occurs during the API request.
        KeyError:
            - Condition: If expected data fields are missing in the API response.
        Exception:
            - Condition: For any other unforeseen errors during execution.
    """

    if not city:
        raise ValueError("City name must be a non-empty string.")

    try:
        async with aiohttp.ClientSession() as session:
            api_url = f"https://api.weatherprovider.com/v1/current?city={city}&apikey=YOUR_API_KEY"
            async with session.get(api_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to retrieve weather data for {city}. Status code: {response.status}")
                    response.raise_for_status()
                data = await response.json()

                temperature = data["temperature"]
                condition = data["condition"]

                weather_info = f"The weather in {city} is {temperature} degrees and {condition}."
                logger.info(f"Retrieved weather for {city}: {weather_info}")
                return weather_info

    except aiohttp.ClientError as e:
        logger.error(f"Network error while fetching weather for {city}: {e}")
        raise
    except KeyError as e:
        logger.error(f"Missing data in API response for {city}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


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
