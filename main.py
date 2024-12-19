import asyncio
import json
from typing import List, Dict, Any, Annotated, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import ToolCallMessage

# from config import OPENAI_API_KEY, MODEL_NAME, DATABASE_PATH
from autogen_core import SingleThreadedAgentRuntime
from autogen_core.models import ModelCapabilities, SystemMessage, UserMessage
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime, GrpcWorkerAgentRuntimeHost
from deepdiff import DeepDiff
from langfuse import Langfuse
from loguru import logger
from pydantic import BaseModel

from agents.ResponseFormatAssistantAgent import ResponseFormatAssistantAgent

# from models.openai_client import OpenAIChatCompletionClient
from agents.coordinator_agent import CoordinatorAgent
from agents.middle_decider_agent import MiddleDeciderAgent
from agents.analysis_agent import AnalysisAgent

from agents.final_pipeline_agent import FinalPipelineAgent
from autogen_core.tool_agent import ToolAgent
from tools.function_tools import (
    # pipeline_a_tool,
    # pipeline_b_tool,
    # final_pipeline_tool,
    DataPipelineAgent,
    add_numbers,
    multiply_numbers,
    # query_tool,
)
from agents.common import UserInput

from utils.settings import settings
from utils.tracing import configure_tracing

# from workers.worker_agent import worker_runtime_client


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
    model_client = OpenAIChatCompletionClient(
        model=settings.model,
        api_key=settings.openai_api_key,
        create_args={"response_format": "json"},
    )

    runtime.start()

    class Filters(BaseModel):
        reason: str
        filters: Dict[str, List[str]] | None
        successful: bool

    def decide_filters(filters_mapped: Annotated[Filters, "Mapped filters from user's query"]) -> Filters:
        return Filters(**filters_mapped.model_dump())

    system_message = (
        "Please map natural language input to filters. Example: what is happening with B35? Answer: 'system': ['B35']"
    )
    # mapping_agent = AssistantAgent(
    #     name="pipeline_selector",
    #     model_client=model_client,
    #     system_message=system_message,
    #     tools=[decide_filters],
    # )

    task_to_result = {
        "why is my gi21 is so bad?": {"reason": Any, "filters": {"system": ["GI21"]}, "successful": True},
        "what is happening with B35?": {"reason": Any, "filters": {"system": ["B35"]}, "successful": True},
        "what is happening with B35 and B36?": {
            "reason": Any,
            "filters": {"system": ["B35", "B36"]},
            "successful": True,
        },
        "what is happening with B35 and B36 and B37?": {
            "reason": Any,
            "filters": {"system": ["B35", "B36", "B37"]},
            "successful": True,
        },
        "halo, barsudjgfsuyfgbsgf": {"reason": Any, "filters": None, "successful": False},
        "CFN*GF &U#FVFS": {"reason": Any, "filters": None, "successful": False},
        "": {"reason": Any, "filters": None, "successful": False},
        "Oppa! Hele, p392756 is on the fly OR WHAT!": {
            "reason": Any,
            "filters": {"system": ["p392756"]},
            "successful": True,
        },
    }

    for task, result_expected in task_to_result.items():
        # # +Structured
        # response = await model_client.create(
        #     messages=mapping_agent._system_messages + [UserMessage(content=task, source="user")],
        #     # tools=[
        #     #     FunctionTool(
        #     #         decide_filters,
        #     #         description=(
        #     #             decide_filters.__doc__
        #     #             if hasattr(decide_filters, "__doc__") and decide_filters.__doc__ is not None
        #     #             else ""
        #     #         ),
        #     #     )
        #     # ],
        #     extra_create_args={"response_format": Filters},
        # )
        #
        # # Ensure the response content is a valid JSON string before loading it
        # response_content: Optional[str] = response.content if isinstance(response.content, str) else None
        # if response_content is None:
        #     raise ValueError("Response content is not a valid JSON string")
        #
        # # Print the response content after loading it as JSON
        # logger.debug(json.loads(response_content))
        #
        # # Validate the response content with the MathReasoning model
        # filters_parsed = Filters.model_validate(json.loads(response_content))
        # # -Structured
        #
        # # result = await mapping_agent.run(task=task)
        # # tool_call_message = next(x for x in result.messages if isinstance(x, ToolCallMessage))
        # # arguments = json.loads(tool_call_message.content[0].arguments)
        #
        # # Identify fields set to Any in the expected result
        # any_fields = [k for k, v in result_expected.items() if v is Any]
        #
        # arguments = filters_parsed.model_dump()
        #
        # # Log the values of the fields that are not checked
        # for field in any_fields:
        #     actual_value = arguments.get(field)
        #     logger.info(f"Field '{field}' is not checked, but its value is: {actual_value}")
        #
        #     # Remove the field from both dictionaries before comparison
        #     result_expected.pop(field, None)
        #     arguments.pop(field, None)
        #
        # # Now compare the remaining fields using DeepDiff
        # diff = DeepDiff(result_expected, arguments)
        # if diff:
        #     logger.error(f"Task: {task}, Differences: {diff}")
        # else:
        #     logger.info(f"Task: {task}, No differences found.")

        agent = ResponseFormatAssistantAgent(
            name="ResponseFormatAssistantAgent",
            model_client=model_client,
            response_format=Filters,
            system_message=system_message,
            tools=[decide_filters],
        )
        result = await agent.run(task=task)
        logger.debug(result)

        i = 100

    i = 100

    # Use python diff utils

    # result = await mapping_agent.run(task="why is my gi21 is so bad?")
    # [TextMessage(source='user', models_usage=None, content='why is my gi21 is so bad?', type='TextMessage'), ToolCallMessage(source='pipeline_selector', models_usage=RequestUsage(prompt_tokens=96, completion_tokens=35), content=[Functi...User is experiencing issues with GI21.","filters":"GI21","successful":true}}', name='decide_filters')], type='ToolCallMessage'), ToolCallResultMessage(source='pipeline_selector', models_usage=None, content=[FunctionExecutionResult(content="Error: 1 valida...n visit https://errors.pydantic.dev/2.10/v/dict_type", call_id='call_qrR3tLVkKk9YjHzhxToYDDbu')], type='ToolCallResultMessage'), TextMessage(source='pipeline_selector', models_usage=None, content="Error: 1 validation error for filters_mapped\nfilters_mapp...e='GI21', input_type=str]\n    For further information visit https://errors.pydantic.dev/2.10/v/dict_type", type='TextMessage')]

    # Need to find FunctionCall.arguments:
    # tool_call_message = next(filter(lambda x: isinstance(x, ToolCallMessage), result.messages))
    # tool_call_message = next(x for x in result.messages if isinstance(x, ToolCallMessage))
    # arguments = json.loads(tool_call_message.content[0].arguments)

    i = 100

    # Set up termination conditions for the assistant agent
    # termination = (
    #     HandoffTermination(target="pipeline_a")
    #     | HandoffTermination(target="pipeline_b")
    #     | HandoffTermination(target="user")
    #     | MaxMessageTermination(5)
    # )

    # # Create a Swarm with the assistant agent
    # team = Swarm(participants=[mapping_agent], termination_condition=termination)
    #
    # # Run the Swarm with the user's task
    # result = await team.run(task=message.text)
    #
    # # Get the last message from the assistant agent
    # last_message = result.messages[-1]

    # # Simulate user input and initiate processing
    # coordinator_agent_id = await runtime.get("coordinator_agent_type", key="default")
    # # user_input_text = input("Enter your request: ")
    # # user_input_text = "Process the data using Pipeline A win wool socks and then finalize the results drinking apple juice"
    # user_input_text = "Process the data using Pipeline A in wool socks and then finalize the results"
    # user_input = UserInput(text=user_input_text)
    #
    # final_result = await runtime.send_message(message=user_input, recipient=coordinator_agent_id)
    #
    # logger.debug("Final Analysis Report:")
    # logger.debug(final_result.result)
    #
    # await runtime.stop()
    # # await worker_host.stop_when_signal()
    # # await worker_runtime_client.stop_when_signal()


if __name__ == "__main__":
    asyncio.run(main())
