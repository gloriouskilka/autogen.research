import json
from typing import List

from autogen_core import SingleThreadedAgentRuntime, FunctionCall
from autogen_core.models import CreateResult
from langfuse import Langfuse
from loguru import logger
import sys

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from pydantic import BaseModel

from util import model_client, settings, configure_tracing

import aiohttp


# Prompt used:
# Write a detailed Function description (it should be a Python code with Python documentation abilities used) explaining parameters in detail, write several examples of parameter values, returned values. This description will be used in a very sensitive context, the logic of a function will be regenerated based on description via LLM, so the function description should be very detailed, not to allow LLM to hallucinate. Limit the description length to 800 symbols, renamed function to have a longer name if needed. No need to focus on async/not async Python implementation details, only focus on input parameters and output.


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
            api_url = f"http://api.weatherstack.com/current?access_key={settings.weatherstack_api_key}&query={city}"
            async with session.get(api_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to retrieve weather data for {city}. Status code: {response.status}")
                    response.raise_for_status()
                data = await response.json()

                temperature = data["current"]["temperature"]
                condition = data["current"]["weather_descriptions"][0]

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


async def main():
    langfuse = Langfuse(
        secret_key=settings.langfuse_secret_key,
        public_key=settings.langfuse_public_key,
        host=settings.langfuse_host,
    )
    logger.info(f"Langfuse host: {langfuse.base_url}")
    logger.info(f"Langfuse project_id: {langfuse.project_id}")

    tracer_provider = configure_tracing(langfuse_client=langfuse)
    runtime = SingleThreadedAgentRuntime(tracer_provider=tracer_provider)

    # Configure loguru
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
        level="DEBUG",
    )

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
    team = RoundRobinGroupChat([weather_agent], termination_condition=text_termination)
    logger.info("Single-agent team created.")

    team._runtime = runtime
    # for t in weather_agent._tools:
    #     t: Tool
    #     logger.info(f"Tool: {t}")
    #     a = t._func
    #     i = 100

    # logger.info("Starting team run.")
    # result = await team.run(task="What is the weather in New York?")
    # logger.info(f"Team run completed with result: {result}")

    model_client.set_throw_on_create(True)

    weather_agent._runtime = runtime

    # TODO: this should be generated, might be cached
    # query_to_agent = "What is the weather in New York?"
    # expected_arguments = {"city": "New York"}

    class QNA(BaseModel):
        query: str
        name: str
        arguments: dict

    test_cases = [
        QNA(
            query="What is the weather in New York?",
            name="get_current_weather_information",
            arguments={"city": "New York"},
        ),
        QNA(
            query="What is the weather in London?", name="get_current_weather_information", arguments={"city": "London"}
        ),
    ]

    state = await weather_agent.save_state()
    for test_case in test_cases:
        try:
            logger.info(f"Running assistant agent with query: {test_case.query}")
            await weather_agent.load_state(state)

            result = await weather_agent.run(task=test_case.query)
            logger.info(f"Agent run completed with result: {result}")

        except model_client.FunctionCallVerification as e:
            logger.info(f"Model completion result: {e.result}")

            assert test_case.name == e.name
            assert test_case.arguments == e.arguments

            logger.info(f"Function called correctly: {e.name} with arguments: {e.arguments}")
        except Exception as e:
            logger.error(f"Error running {e}")
        finally:
            logger.info("Completed test case.")
            weather_agent.


# If you're running this script directly, you can use asyncio to run the async function
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
