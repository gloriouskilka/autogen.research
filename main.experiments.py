import asyncio
import json
import sys

from autogen_agentchat.agents import UserProxyAgent, AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console

# from config import OPENAI_API_KEY, MODEL_NAME, DATABASE_PATH
from autogen_core import SingleThreadedAgentRuntime, CancellationToken, MessageContext

# from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.runtimes.grpc import GrpcWorkerAgentRuntime, GrpcWorkerAgentRuntimeHost
from langfuse import Langfuse
from loguru import logger

from _old.test_utils.prev.mock_chat_client import MockChatCompletionClient

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
from utils.test_utils import OpenAIChatCompletionClientWrapper
from utils.tracing import configure_tracing

# from workers.worker_agent import worker_runtime_client


# Define a custom function to handle verification exceptions
async def handle_verification(verification, expected_function_calls):
    if isinstance(verification, OpenAIChatCompletionClientWrapper.FunctionCallVerification):
        # Handle function call verification
        actual_function_calls = []
        for function_call_record in verification.function_calls:
            function_name = function_call_record.function_name
            arguments = function_call_record.arguments
            print(f"Function called: {function_name} with arguments: {arguments}")
            actual_function_calls.append({"name": function_name, "arguments": arguments})

        # Compare actual function calls with expected function calls
        if actual_function_calls == expected_function_calls:
            print("Function calls match the expected function calls.")
        else:
            print("Function calls do not match the expected function calls.")
            print("Expected:")
            print(json.dumps(expected_function_calls, indent=2))
            print("Actual:")
            print(json.dumps(actual_function_calls, indent=2))
    elif isinstance(verification, OpenAIChatCompletionClientWrapper.TextResultVerification):
        # Handle text result verification
        content = verification.content
        print(f"Text content: {content}")
        # Implement your business logic based on the text content
    else:
        # Handle unexpected verification types
        raise Exception("Unknown verification type.")


async def main():
    model_client = OpenAIChatCompletionClientWrapper(
        model="gpt-4o-mini",
        api_key=settings.openai_api_key,
    )

    langfuse = Langfuse(
        secret_key=settings.langfuse_secret_key,
        public_key=settings.langfuse_public_key,
        host=settings.langfuse_host,
    )
    logger.info(f"Langfuse host: {langfuse.base_url}")
    logger.info(f"Langfuse project_id: {langfuse.project_id}")

    tracer_provider = configure_tracing(langfuse_client=langfuse)
    runtime = SingleThreadedAgentRuntime(tracer_provider=tracer_provider)

    await CoordinatorAgent.register(
        runtime=runtime,
        type="coordinator_agent_type",
        factory=lambda: CoordinatorAgent(model_client),
    )

    await MiddleDeciderAgent.register(
        runtime=runtime,
        type="middle_decider_agent_type",
        factory=lambda: MiddleDeciderAgent(
            model_client=model_client,
            description="Analyse the data and write a summary for the user - what to do to improve the results",  # TODO: is it used somewhere?
            system_message_summarizer="Analyse the data and write a summary for the user",
            tools=[add_numbers, multiply_numbers],
        ),
    )
    await AnalysisAgent.register(
        runtime=runtime, type="analysis_agent_type", factory=lambda: AnalysisAgent(model_client)
    )
    await FinalPipelineAgent.register(runtime=runtime, type="final_pipeline_agent_type", factory=FinalPipelineAgent)

    await DataPipelineAgent.register(
        runtime,
        "data_pipeline_agent",
        lambda: DataPipelineAgent(),
    )

    # coordinator_agent_id = await runtime.get("coordinator_agent_type", key="default")
    # user_input_text = "Process the data using Pipeline A in wool socks and then finalize the results"
    # user_input = UserInput(text=user_input_text)
    #
    # final_result = await runtime.send_message(message=user_input, recipient=coordinator_agent_id)
    #
    # logger.debug("Final Analysis Report:")
    # logger.debug(final_result.result)
    #
    # await runtime.stop()

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

    # model_client = OpenAIChatCompletionClient(model="gpt-4o")

    # Define some tools, for example:
    # async def tool_example() -> str:
    #     """A simple example tool."""
    #     return "Example tool result"

    # Using AssistantAgent (Option 1)
    # agent = AssistantAgent(name="assistant", model_client=model_client, tools=[tool_example])
    # agent = CoordinatorAgent(model_client=model_client)

    # agent = CoordinatorAgent(model_client)
    #
    # coordinator_agent_id = await runtime.get("coordinator_agent_type", key="default")
    # user_input_text = "Process the data using Pipeline A in wool socks and then finalize the results"
    # user_input = UserInput(text=user_input_text)

    # # Define your task and execute
    # cancellation_token = CancellationToken()
    # response = await agent..send_message(
    #     [(TextMessage(content="Perform a task using tools.", source="user"))], cancellation_token
    # )
    # logger.debug(response.chat_message)

    # Initialize the database with sample data
    # await init_db()

    # ---------------------------
    # Team Setup (Swarm)
    # ---------------------------

    # team = Swarm(
    #     participants=[
    #         planner_agent,
    #         sql_query_agent,
    #         data_analysis_agent,
    #         explanation_agent,
    #     ],
    #     termination_condition=termination_condition,
    # )

    # await MiddleDeciderAgent.register(
    #     runtime=runtime,
    #     type="middle_decider_agent_type",
    #     factory=lambda: MiddleDeciderAgent(
    #         model_client=model_client,
    #         description="Analyse the data and write a summary for the user - what to do to improve the results",  # TODO: is it used somewhere?
    #         system_message_summarizer="Analyse the data and write a summary for the user",
    #         tools=[add_numbers, multiply_numbers],
    #     ),
    # )

    agent: CoordinatorAgent = await runtime.try_get_underlying_agent_instance(
        id="coordinator_agent_type", type=CoordinatorAgent
    )
    assert isinstance(agent, CoordinatorAgent)

    # agent._runtime = runtime
    tasks = [
        {
            "task": "Investigate why Black Hat is becoming more popular in our shop.",
            "expected_function_calls": [
                {"name": "get_sales_data", "arguments": {"product_name": "Black Hat"}},
                {"name": "get_customer_feedback", "arguments": {"product_name": "Black Hat"}},
            ],
        },
        {
            "task": "Investigate why Yellow Hat is not popular in our shop.",
            "expected_function_calls": [
                {"name": "get_sales_data", "arguments": {"product_name": "Yellow Hat"}},
                {"name": "get_customer_feedback", "arguments": {"product_name": "Yellow Hat"}},
            ],
        },
    ]
    cancellation_token = CancellationToken()

    model_client.set_throw_on_create(True)

    # saved_state = await agent.save_state()

    # coordinator_agent_id = await runtime.get("coordinator_agent_type", key="default")
    # user_input_text = input("Enter your request: ")
    # user_input_text = "Process the data using Pipeline A win wool socks and then finalize the results drinking apple juice"
    user_input_text = "Process the data using Pipeline A in wool socks and then finalize the results"
    user_input = UserInput(text=user_input_text)

    runtime.start()

    # Run the tasks sequentially
    for task_entry in tasks:
        task_text = task_entry["task"]
        expected_function_calls = task_entry["expected_function_calls"]

        logger.debug(f"--- Starting task: {task_text} ---\n")

        # MessageContext:
        #     sender: AgentId | None
        #     topic_id: TopicId | None
        #     is_rpc: bool
        #     cancellation_token: CancellationToken
        #     message_id: str

        try:
            # Run the team and capture the output
            # await Console(agent.run_stream(task=task_text))

            result = await agent.handle_user_input(
                message=user_input,
                ctx=MessageContext(
                    sender=None,
                    topic_id=None,
                    is_rpc=False,
                    cancellation_token=cancellation_token,
                    message_id="",
                ),
            )

            # final_result = await runtime.send_message(message=user_input, recipient=coordinator_agent_id)
            i = 100

            # final_result = await runtime.send_message(message=user_input, recipient=coordinator_agent_id)

            # TODO: to improve case when the result is also compared
            # # After running, check the verification results
            # for verification in model_client.create_results:
            #     await handle_verification(verification)
            #
            # # Clear the create_results after handling
            # model_client.create_results.clear()

        except OpenAIChatCompletionClientWrapper.Verification as verification:
            await handle_verification(verification, expected_function_calls)

        logger.debug(f"\n--- Completed task: {task_text} ---\n")
        # await team.reset()  # Reset the team state before each task
        # await agent.load_state(saved_state)
    i = 100


if __name__ == "__main__":
    asyncio.run(main())
