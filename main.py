# main.py

import asyncio
from autogen_core import SingleThreadedAgentRuntime
from autogen_ext.models.openai import OpenAIChatCompletionClient
from langfuse import Langfuse
from loguru import logger

# from models.llm_client import OpenAIChatCompletionClient
from agents.coordinator_agent import CoordinatorAgent
from agents.pipeline_a_agent import PipelineAAgent
from agents.middle_decider_agent import MiddleDeciderAgent
from agents.final_pipeline_agent import FinalPipelineAgent
from agents.analysis_agent import AnalysisAgent
from autogen_core.tool_agent import ToolAgent
from tools.data_tools import pipeline_a_tool, pipeline_b_tool
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
    # runtime = SingleThreadedAgentRuntime(tracer_provider=tracer_provider)
    model_client = OpenAIChatCompletionClient(model="gpt-4o", api_key=settings.openai_api_key)

    # Register agents
    await CoordinatorAgent.register(runtime, "coordinator_agent", lambda: CoordinatorAgent(model_client))
    await PipelineAAgent.register(runtime, "pipeline_a_agent", PipelineAAgent)
    await MiddleDeciderAgent.register(runtime, "middle_decider_agent", lambda: MiddleDeciderAgent(model_client))
    await FinalPipelineAgent.register(runtime, "final_pipeline_agent", FinalPipelineAgent)
    await AnalysisAgent.register(runtime, "analysis_agent", lambda: AnalysisAgent(model_client))

    # Register ToolAgent
    tool_agent = ToolAgent(description="Tool Agent", tools=[pipeline_a_tool, pipeline_b_tool])
    await tool_agent.register(runtime, "tool_agent", lambda: tool_agent)

    runtime.start()

    # Get user input
    user_input = input("Enter your request: ")

    # Start processing
    coordinator_agent_id = await runtime.get("coordinator_agent")
    await runtime.send_message(message={"text": user_input}, recipient=coordinator_agent_id)


if __name__ == "__main__":
    asyncio.run(main())
