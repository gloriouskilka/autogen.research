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


async def get_current_weather_information(city: str) -> str:
    """
    Retrieves the current weather information for a specified city using an external weather API.

    Parameters:
    city (str): The name of the city for which to obtain weather data.
                - Must be a non-empty string.
                - Example values: "London", "San Francisco", "Tokyo".

    Returns:
    str: A human-readable string describing the current weather in the specified city.
         - Includes temperature and weather conditions.
         - Example return values:
             - "The weather in London is 15 degrees and rainy."
             - "The weather in San Francisco is 22 degrees and sunny."
             - "The weather in Tokyo is 18 degrees and cloudy."

    Raises:
    ValueError: If the `city` parameter is empty or not provided.
    Exception: For any network issues, API errors, or unexpected problems during data retrieval.

    Notes:
    - The function interacts with an external API endpoint:
      "https://api.weatherprovider.com/v1/current".
    - An API key is required for authentication (replace 'YOUR_API_KEY' with a valid key).
    - Ensure proper error handling when integrating this function into applications.

    Examples:
    >>> get_current_weather_information("Berlin")
    "The weather in Berlin is 20 degrees and clear."

    >>> get_current_weather_information("Sydney")
    "The weather in Sydney is 25 degrees and sunny."
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
    tools=[get_current_weather_information],
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
