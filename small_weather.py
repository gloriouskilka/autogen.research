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
    Asynchronously retrieves the current weather information for a specified city.

    This function connects to a weather API to fetch the latest weather data for the given city.
    It processes the response and returns a formatted string containing the temperature and weather condition.
    The function is designed to be used in environments where non-blocking I/O operations are preferred.

    Parameters
    ----------
    city : str
        The name of the city for which to retrieve the weather information.

        **Requirements:**
        - Must be a non-empty string.
        - Should correspond to a valid city recognized by the weather API.
        - Case-insensitive but typically capitalized (e.g., "New York", "los angeles").
        - Examples of valid values:
            - "New York"
            - "London"
            - "Tokyo"
            - "São Paulo"
            - "Sydney"

    Returns
    -------
    str
        A formatted string detailing the current weather in the specified city.

        **Format:**
        `"The weather in {city} is {temperature} degrees and {condition}."`

        **Components:**
        - `{city}`: The input city name.
        - `{temperature}`: The current temperature in degrees Fahrenheit.
        - `{condition}`: A brief description of the current weather condition (e.g., "Sunny", "Rainy").

        **Examples of returned values:**
        - "The weather in New York is 72 degrees and Sunny."
        - "The weather in London is 58 degrees and Cloudy."
        - "The weather in Tokyo is 85 degrees and Humid."

    Raises
    ------
    ValueError
        If the `city` parameter is an empty string.
    aiohttp.ClientError
        If there is an issue with the network request to the weather API.
    KeyError
        If the expected data fields are missing from the API response.
    Exception
        For any other unforeseen errors during execution.

    Examples
    --------
    Example 1: Valid city input
    >>> import asyncio
    >>> asyncio.run(get_weather("New York"))
    "The weather in New York is 72 degrees and Sunny."

    Example 2: Another valid city input
    >>> asyncio.run(get_weather("London"))
    "The weather in London is 58 degrees and Cloudy."

    Example 3: Handling invalid city input
    >>> asyncio.run(get_weather(""))
    Traceback (most recent call last):
        ...
    ValueError: City name must be a non-empty string.

    Notes
    -----
    - This function uses asynchronous HTTP requests to prevent blocking the event loop.
    - Ensure that the event loop is properly managed when integrating this function into larger applications.
    - The temperature is assumed to be in degrees Fahrenheit; adjust the API endpoint or processing logic if a different unit is required.

    Security Considerations
    -----------------------
    - Ensure that the city names are properly sanitized if they are sourced from user input to prevent injection attacks.
    - Securely manage API keys or tokens required for accessing the weather service API.

    Dependencies
    ------------
    - `aiohttp`: For making asynchronous HTTP requests.
    - `logging`: For logging information and debugging purposes.

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
