import asyncio
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Annotated, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import ToolCallMessage

# from config import OPENAI_API_KEY, MODEL_NAME, DATABASE_PATH
from autogen_core import SingleThreadedAgentRuntime
from autogen_core.models import ModelCapabilities, SystemMessage, UserMessage
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

# from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime, GrpcWorkerAgentRuntimeHost
from deepdiff import DeepDiff
from langfuse import Langfuse
from loguru import logger
from openai.types import FunctionDefinition, FunctionParameters
from pydantic import BaseModel, Field, ConfigDict

from agents.ResponseFormatAssistantAgent import ResponseFormatAssistantAgent

# from models.openai_client import OpenAIChatCompletionClient
from agents.coordinator_agent import CoordinatorAgent
from agents.middle_decider_agent import MiddleDeciderAgent
from agents.analysis_agent import AnalysisAgent

from agents.final_pipeline_agent import FinalPipelineAgent
from autogen_core.tool_agent import ToolAgent

from models.openai_client import (
    MyOpenAIChatCompletionClient,
    OpenAIChatCompletionClientWrapper,
)
from tools.function_tools import (
    # pipeline_a_tool,
    # pipeline_b_tool,
    # final_pipeline_tool,
    DataPipelineAgent,
    add_numbers,
    multiply_numbers,
    # query_tool,
)
from agents.common import UserInput, Filters, FilterItem, FiltersReflect

from utils.settings import settings
from utils.tracing import configure_tracing

# from workers.worker_agent import worker_runtime_client


# +Prior check code, should be incorporated into the main code below
#         any_fields = [k for k, v in result_expected.items() if v is Any]
#
#         arguments = filters_parsed.model_dump()
#
#         # Log the values of the fields that are not checked
#         for field in any_fields:
#             actual_value = arguments.get(field)
#             logger.info(
#                 f"Field '{field}' is not checked, but its value is: {actual_value}"
#             )
#
#             # Remove the field from both dictionaries before comparison
#             result_expected.pop(field, None)
#             arguments.pop(field, None)
#
#         # Now compare the remaining fields using DeepDiff
#         diff = DeepDiff(result_expected, arguments)
#         if diff:
#             logger.error(f"Task: {task}, Differences: {diff}")
#         else:
#             logger.info(f"Task: {task}, No differences found.")
#
#         logger.debug(result)
# -Prior check code, should be incorporated into the main code below


async def handle_verification(verification, expected_function_calls):
    if isinstance(
        verification, OpenAIChatCompletionClientWrapper.FunctionCallVerification
    ):
        # Handle function call verification
        actual_function_calls = []
        for function_call_record in verification.function_calls:
            function_name = function_call_record.function_name
            arguments = function_call_record.arguments
            print(f"Function called: {function_name} with arguments: {arguments}")
            actual_function_calls.append(
                {"function_name": function_name, "arguments": arguments}
            )

        # Compare actual function calls with expected function calls
        any_fields = [
            k
            for k, v in expected_function_calls[0]["arguments"]["filters"].items()
            if v is Any
        ]

        # Log the values of the fields that are not checked
        for field in any_fields:
            actual_value = actual_function_calls[0]["arguments"]["filters"].get(field)
            logger.info(
                f"Field '{field}' is not checked, but its value is: {actual_value}"
            )

            # Remove the field from both dictionaries before comparison
            expected_function_calls[0]["arguments"]["filters"].pop(field, None)
            actual_function_calls[0]["arguments"]["filters"].pop(field, None)

        # Now compare the remaining fields using DeepDiff
        diff = DeepDiff(expected_function_calls, actual_function_calls)
        if diff:
            logger.error(f"Differences: {diff}")
        else:
            logger.info("No differences found.")

    elif isinstance(
        verification, OpenAIChatCompletionClientWrapper.TextResultVerification
    ):
        # Handle text result verification
        content = verification.content
        print(f"Text content: {content}")
        # Implement your business logic based on the text content
    else:
        # Handle unexpected verification types
        raise Exception("Unknown verification type.")


# async def handle_verification_previous(verification, expected_function_calls):
#     if isinstance(
#         verification, OpenAIChatCompletionClientWrapper.FunctionCallVerification
#     ):
#         # Handle function call verification
#         actual_function_calls = []
#         for function_call_record in verification.function_calls:
#             function_name = function_call_record.function_name
#             arguments = function_call_record.arguments
#             print(f"Function called: {function_name} with arguments: {arguments}")
#             actual_function_calls.append(
#                 {"function_name": function_name, "arguments": arguments}
#             )
#
#         # Compare actual function calls with expected function calls
#         if actual_function_calls == expected_function_calls:
#             print("Function calls match the expected function calls.")
#         else:
#             print("Function calls do not match the expected function calls.")
#             print("Expected:")
#             print(json.dumps(expected_function_calls, indent=2))
#             print("Actual:")
#             print(json.dumps(actual_function_calls, indent=2))
#     elif isinstance(
#         verification, OpenAIChatCompletionClientWrapper.TextResultVerification
#     ):
#         # Handle text result verification
#         content = verification.content
#         print(f"Text content: {content}")
#         # Implement your business logic based on the text content
#     else:
#         # Handle unexpected verification types
#         raise Exception("Unknown verification type.")


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
    # model_client = OpenAIChatCompletionClient(  # ERROR: ValueError: `decide_filters` is not strict. Only `strict` function tools can be auto-parsed
    #     model=settings.model,
    #     api_key=settings.openai_api_key,
    # )

    # model_client = MyOpenAIChatCompletionClient(  # This has a fix
    model_client = OpenAIChatCompletionClientWrapper(  # This has a fix
        model=settings.model,
        api_key=settings.openai_api_key,
    )

    runtime.start()

    # , cancellation_token: CancellationToken
    def decide_system_filters(
        filters: Annotated[Filters, "Mapped filters"],
    ) -> Filters:
        """
        The user will ask about a system or multiple systems, no need to worry what that means.
        Please map the user's input to the system filters. Key always must be 'system'.
        Examples: "what is happening with B35 and B36 and B37?" -> {"system": ["B35", "B36", "B37"]}
        """

        filters_dict = filters.model_dump() if isinstance(filters, Filters) else filters
        assert isinstance(filters_dict, dict)

        return Filters(**filters_dict)

    from autogen_core._function_utils import get_function_schema

    func_schema = get_function_schema(
        decide_system_filters, description=decide_system_filters.__doc__
    )
    i = 100

    # task_to_result = {
    #     "why is my gi21 is so bad?": {"reason": Any, "filters": {"system": ["GI21"]}, "successful": True},
    #     "what is happening with B35?": {"reason": Any, "filters": {"system": ["B35"]}, "successful": True},
    #     "what is happening with B35 and B36?": {
    #         "reason": Any,
    #         "filters": {"system": ["B35", "B36"]},
    #         "successful": True,
    #     },
    #     "what is happening with B35 and B36 and B37?": {
    #         "reason": Any,
    #         "filters": {"system": ["B35", "B36", "B37"]},
    #         "successful": True,
    #     },
    #     "halo, barsudjgfsuyfgbsgf": {"reason": Any, "filters": None, "successful": False},
    #     "CFN*GF &U#FVFS": {"reason": Any, "filters": None, "successful": False},
    #     "": {"reason": Any, "filters": None, "successful": False},
    #     "Oppa! Hele, p392756 is on the fly OR WHAT!": {
    #         "reason": Any,
    #         "filters": {"system": ["p392756"]},
    #         "successful": True,
    #     },
    # }

    # The format was changed, this is correct one: {'filters': [{'key': 'GI21_issues', 'values': ['performance', 'feedback', 'training', 'environment', 'stress', 'objectives']}], 'successful': True}

    # task_to_result = {
    #     "why is my gi21 is so bad?": {
    #         "reason": Any,
    #         "filters": [
    #             {
    #                 "key": "system",
    #                 "values": ["gi21"],
    #             }
    #         ],
    #         "successful": True,
    #     },
    # }
    # Need to add info about function name:
    # # FunctionCallRecord(function_name='decide_system_filters', arguments={'filters': {'reason': "Extract system IDs from the user's query.", 'filters': [{'key': 'System IDs', 'values': ['gi21']}], 'successful': True}})

    task_to_result = {
        "why is my gi21 is so bad?": [
            {
                "function_name": "decide_system_filters",
                "arguments": {
                    "filters": {
                        "reason": Any,
                        "filters": [{"key": "system", "values": ["gi21"]}],
                        "successful": True,
                    }
                },
            }
        ]
    }

    model_client.set_throw_on_create(True)

    agent = ResponseFormatAssistantAgent(
        name="ResponseFormatAssistantAgent",
        model_client=model_client,
        response_format=Filters,
        system_message="The user will mention some IDs - those IDs are system's names, please help to extract them",
        tools=[FunctionTool(decide_system_filters, description="DAVAI")],
        # reflect_on_tool_use=True,
        # response_format_reflect_on_tool_use=FiltersReflect,
    )

    for task, expected_function_calls in task_to_result.items():
        # +intercept
        saved_state = await agent.save_state()

        # expected_function_calls = []

        try:
            # result = await agent.run(task=task)
            result: TaskResult = await agent.run(task=task)
            i = 100

            # Run the tasks sequentially
            # expected_function_calls = task_entry["expected_function_calls"]
            # Run the team and capture the output
            # await Console(team.run_stream(task=task_text))

            # After running, check the verification results
            # for verification in model_client.create_results:
            #     await handle_verification(verification)

            # Clear the create_results after handling
            # model_client.create_results.clear()  # if set_throw_on_create wasn't set

        except OpenAIChatCompletionClientWrapper.Verification as verification:
            await handle_verification(verification, expected_function_calls)

        # logger.debug(f"\n--- Completed task: {task_text} ---\n")
        # await team.reset()  # Reset the team state before each task
        await agent.load_state(saved_state)
        # -intercept

        # +Before set_throw_on_create=True

        # # result = await agent.run(task=task)
        # result: TaskResult = await agent.run(task=task)
        # response_content = result.messages[-1].content
        # # Ensure the response content is a valid JSON string before loading it
        # # response_content: Optional[str] = response.content if isinstance(response.content, str) else None
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
        #     logger.info(
        #         f"Field '{field}' is not checked, but its value is: {actual_value}"
        #     )
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
        #
        # logger.debug(result)
        # -Before set_throw_on_create=True

        i = 100
        break

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
