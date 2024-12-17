from langfuse import Langfuse

from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.ui import Console
from autogen_core import SingleThreadedAgentRuntime
from langfuse.decorators import observe
from loguru import logger
import sys

# Define a tool that gets the weather for a city.
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, HandoffTermination
from autogen_agentchat.teams import Swarm

from util import model_client, settings, configure_tracing

from opentelemetry import trace


# @observe
def test_function():
    with trace.get_tracer(__name__).start_as_current_span("test-span") as span:
        logger.debug("This is a final test span.")


@observe
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

    agent = AssistantAgent(
        "NotAlice",
        model_client=model_client,
        handoffs=["user"],
        system_message="You are NotAlice and you only answer questions about yourself, ask the user for help if needed.",
    )
    termination = HandoffTermination(target="user") | MaxMessageTermination(3)
    team = Swarm([agent], termination_condition=termination)

    team._runtime = runtime
    # runtime.start()

    # Start the conversation.
    await Console(team.run_stream(task="What is bob's birthday?"))

    # Resume with user feedback.
    await Console(
        team.run_stream(
            task=HandoffMessage(source="user", target="NotAlice", content="Bob's birthday is on 1st January.")
        )
    )

    test_function()


# If you're running this script directly, you can use asyncio to run the async function
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
