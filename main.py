import asyncio

# from config import OPENAI_API_KEY, MODEL_NAME, DATABASE_PATH
from autogen_core import SingleThreadedAgentRuntime
from autogen_ext.models.openai import OpenAIChatCompletionClient
from langfuse import Langfuse
from loguru import logger

# from models.openai_client import OpenAIChatCompletionClient
from agents.coordinator_agent import CoordinatorAgent
from agents.middle_decider_agent import MiddleDeciderAgent
from agents.analysis_agent import AnalysisAgent

# from agents.pipeline_a_agent import PipelineAAgent
# from agents.pipeline_b_agent import PipelineBAgent
from agents.final_pipeline_agent import FinalPipelineAgent
from autogen_core.tool_agent import ToolAgent
from tools.function_tools import (
    pipeline_a_tool,
    pipeline_b_tool,
    final_pipeline_tool,
    # query_tool,
)
from agents.common import UserInput

from utils.settings import settings
from utils.tracing import configure_tracing


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
    model_client = OpenAIChatCompletionClient(model=settings.model, api_key=settings.openai_api_key)

    # Register agents with specific keys
    await CoordinatorAgent.register(
        runtime=runtime,
        type="coordinator_agent_type",
        factory=lambda: CoordinatorAgent(model_client),
    )
    await MiddleDeciderAgent.register(
        runtime=runtime,
        type="middle_decider_agent_type",
        factory=lambda: MiddleDeciderAgent(model_client),
    )
    await AnalysisAgent.register(
        runtime=runtime, type="analysis_agent_type", factory=lambda: AnalysisAgent(model_client)
    )
    await FinalPipelineAgent.register(runtime=runtime, type="final_pipeline_agent_type", factory=FinalPipelineAgent)

    await ToolAgent.register(
        runtime,
        "tool_agent_type",
        lambda: ToolAgent(
            description="Pipeline Tool Agent",
            tools=[pipeline_a_tool, pipeline_b_tool, final_pipeline_tool],
        ),
    )

    runtime.start()

    # Simulate user input and initiate processing
    coordinator_agent_id = await runtime.get("coordinator_agent_type")
    # user_input_text = input("Enter your request: ")
    user_input_text = "Process the data using Pipeline A and then finalize the results"
    user_input = UserInput(text=user_input_text)

    final_result = await runtime.send_message(message=user_input, recipient=coordinator_agent_id)

    logger.debug("Final Analysis Report:")
    logger.debug(final_result.result)

    await runtime.stop()


if __name__ == "__main__":
    asyncio.run(main())
