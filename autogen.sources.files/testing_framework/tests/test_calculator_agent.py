# =============================================================================
# tests/test_calculator_agent.py
# =============================================================================

# Example test case for the CalculatorAgent using the MockChatCompletionClient.

import asyncio
from autogen_core import SingleThreadedAgentRuntime, AgentId
from autogen_core.tool_agent import tool_agent_caller_loop
from autogen_core.models import SystemMessage, UserMessage
from test_utils.mock_chat_client import MockChatCompletionClient
from test_utils.assertions import assert_function_call
from tools.math_tools import AddNumbersTool
from agents.calculator_agent import register_calculator_agent


async def test_calculator_agent():
    # Initialize the runtime and register the agent
    runtime = SingleThreadedAgentRuntime()
    await register_calculator_agent(runtime)

    # Create a mock chat client
    chat_client = MockChatCompletionClient()

    # Set expected function calls
    function_call = FunctionCall(id="1", name="add_numbers", arguments=json.dumps({"x": 3, "y": 4}))
    chat_client.set_expected_function_calls([function_call])

    # Start the runtime
    runtime.start()

    # Define the messages
    messages = [
        SystemMessage(content="You are a calculator assistant."),
        UserMessage(content="What is 3 plus 4?", source="user"),
    ]

    # Define the tool schema
    add_numbers_tool = AddNumbersTool()
    tool_schema = [add_numbers_tool.schema]

    # Run the tool agent caller loop
    generated_messages = await tool_agent_caller_loop(
        caller=runtime,
        tool_agent_id=AgentId(type="CalculatorAgent", key="default"),
        model_client=chat_client,
        input_messages=messages,
        tool_schema=tool_schema,
        cancellation_token=CancellationToken(),
        caller_source="assistant",
    )

    # Stop the runtime
    await runtime.stop_when_idle()

    # Assertions

    # Assert that the function call was made correctly
    function_calls = chat_client.get_function_calls()
    assert_function_call(
        function_calls,
        expected_name="add_numbers",
        expected_arguments={"x": 3, "y": 4},
    )

    # Assert that the response is correct
    # Since we did not set a specific response, we can check the function execution result
    assert generated_messages[-1].content == [
        {"content": "7.0", "call_id": "1"}
    ], "Unexpected function execution result."
    print("Test passed: CalculatorAgent correctly executed add_numbers function.")


if __name__ == "__main__":
    asyncio.run(test_calculator_agent())
