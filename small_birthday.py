import os

from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.ui import Console
from autogen_core import SingleThreadedAgentRuntime
from langfuse.decorators import observe
from loguru import logger
import sys

# Define a tool that gets the weather for a city.
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination, HandoffTermination
from autogen_agentchat.teams import RoundRobinGroupChat, Swarm
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from util import model_client, settings

import aiohttp


from opentelemetry import trace

# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


# def configure_oltp_tracing(endpoint: str = None) -> trace.TracerProvider:
#     # Configure Tracing
#     tracer_provider = TracerProvider(resource=Resource({"service.name": "my-service"}))
#     processor = BatchSpanProcessor(OTLPSpanExporter())
#     tracer_provider.add_span_processor(processor)
#     trace.set_tracer_provider(tracer_provider)
#
#     return tracer_provider


# Prompt used:
# Write a detailed Function description (it should be a Python code with Python documentation abilities used) explaining parameters in detail, write several examples of parameter values, returned values. This description will be used in a very sensitive context, the logic of a function will be regenerated based on description via LLM, so the function description should be very detailed, not to allow LLM to hallucinate. Limit the description length to 800 symbols, renamed function to have a longer name if needed. No need to focus on async/not async Python implementation details, only focus on input parameters and output.


@observe
async def main():
    logger.debug(f"Settings: {settings}")
    # settings
    # tracer_provider = configure_oltp_tracing()

    # +Copy-paste
    # runtime = SingleThreadedAgentRuntime(tracer_provider=tracer_provider)

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

    agent = AssistantAgent(
        "Alice",
        model_client=model_client,
        handoffs=["user"],
        system_message="You are Alice and you only answer questions about yourself, ask the user for help if needed.",
    )
    termination = HandoffTermination(target="user") | MaxMessageTermination(3)
    team = Swarm([agent], termination_condition=termination)

    # team._runtime = runtime
    # runtime.start()

    # Start the conversation.
    await Console(team.run_stream(task="What is bob's birthday?"))

    # Resume with user feedback.
    await Console(
        team.run_stream(task=HandoffMessage(source="user", target="Alice", content="Bob's birthday is on 1st January."))
    )

    # # Define a termination condition.
    # text_termination = TextMentionTermination("TERMINATE")
    # logger.info("Termination condition defined.")
    #
    # # Create a single-agent team.
    # # single_agent_team = RoundRobinGroupChat([weather_agent], termination_condition=text_termination)
    # logger.info("Single-agent team created.")
    #
    # logger.info("Starting team run.")
    # # result = await single_agent_team.run(task="What is the weather in New York?")
    # logger.info(f"Team run completed with result: {result}")


# If you're running this script directly, you can use asyncio to run the async function
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
