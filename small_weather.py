import os

from autogen_core import SingleThreadedAgentRuntime
from loguru import logger
import sys

# Define a tool that gets the weather for a city.
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from util import model_client

import aiohttp


from opentelemetry import trace

# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def configure_oltp_tracing(endpoint: str = None) -> trace.TracerProvider:
    # Configure Tracing
    tracer_provider = TracerProvider(resource=Resource({"service.name": "my-service"}))
    processor = BatchSpanProcessor(OTLPSpanExporter())
    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)

    return tracer_provider


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


async def main():
    tracer_provider = configure_oltp_tracing()

    # +Copy-paste
    # runtime = SingleThreadedAgentRuntime()
    runtime = SingleThreadedAgentRuntime(tracer_provider=tracer_provider)

    # Create an appropriate client
    # client = create_completion_client_from_env()

    # # Register agents.
    # await MultimodalWebSurfer.register(
    #     runtime,
    #     "WebSurfer",
    #     lambda: MultimodalWebSurfer(),
    # )
    # web_surfer = AgentProxy(AgentId("WebSurfer", "default"), runtime)
    #
    # await UserProxy.register(
    #     runtime,
    #     "UserProxy",
    #     lambda: UserProxy(),
    # )
    # user_proxy = AgentProxy(AgentId("UserProxy", "default"), runtime)
    # await RoundRobinOrchestrator.register(
    #     runtime, "orchestrator", lambda: RoundRobinOrchestrator([web_surfer, user_proxy])
    # )

    # actual_surfer = await runtime.try_get_underlying_agent_instance(web_surfer.id, MultimodalWebSurfer)
    # await actual_surfer.init(
    #     model_client=client, downloads_folder=os.getcwd(), browser_channel="chromium", debug_dir=DEBUG_DIR
    # )
    #
    # await runtime.send_message(
    #     BroadcastMessage(
    #         content=UserMessage(
    #             content="Please visit the page 'https://en.wikipedia.org/wiki/Microsoft'", source="user"
    #         )
    #     ),
    #     recipient=web_surfer.id,
    #     sender=user_proxy.id,
    # )
    # -Copy-paste

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

    # Create an assistant agent.
    weather_agent = AssistantAgent(
        "assistant",
        model_client=model_client,
        tools=[get_current_weather_information],
        system_message="Respond 'TERMINATE' when task is complete.",
    )
    logger.info("Assistant agent created.")

    runtime.start()

    # Define a termination condition.
    text_termination = TextMentionTermination("TERMINATE")
    logger.info("Termination condition defined.")

    # Create a single-agent team.
    # single_agent_team = RoundRobinGroupChat([weather_agent], termination_condition=text_termination)
    logger.info("Single-agent team created.")

    logger.info("Starting team run.")
    # result = await single_agent_team.run(task="What is the weather in New York?")
    logger.info(f"Team run completed with result: {result}")


# If you're running this script directly, you can use asyncio to run the async function
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
