# testing_framework_example.py: Testing Agents and Tools in Autogen Framework

# This example demonstrates how to create a testing system for verifying the stability
# and correctness of agent interactions with LLMs, specifically focusing on function
# calls and parameter passing. It includes a custom test client that simulates the
# OpenAI API, intercepts function tool calls, and verifies the function names and parameters.

# =============================================================================
# Overview
# =============================================================================

# The testing framework consists of:
# - A mock ChatCompletionClient that simulates the behavior of the OpenAI LLM.
# - Interceptors that capture and validate function calls and arguments.
# - Utilities for asserting expected behaviors.
# - Example agents and tools to demonstrate testing usage.

# =============================================================================
# test_utils/mock_chat_client.py
# =============================================================================

# A mock implementation of ChatCompletionClient that allows interception and validation
# of function calls.

from autogen_core.models import ChatCompletionClient, CreateResult, LLMMessage, FunctionExecutionResultMessage
from autogen_core._types import FunctionCall
from autogen_core.models import AssistantMessage, FunctionExecutionResult
from autogen_core.tools import Tool, ToolSchema
from typing import List, Optional, Mapping, Any
from autogen_core import CancellationToken
import json


class MockChatCompletionClient(ChatCompletionClient):
    def __init__(self):
        self.function_calls = []
        self.responses = []
        self.expected_calls = []

    async def create(
        self,
        messages: List[LLMMessage],
        tools: List[Tool | ToolSchema] = [],
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        # Simulate the LLM determining to call a function/tool
        # For testing, we can predefine expected function calls
        if self.expected_calls:
            function_call = self.expected_calls.pop(0)
        else:
            # Default behavior: Echo the user's message
            content = messages[-1].content if messages else ""
            return CreateResult(
                finish_reason="stop",
                content=content,
                usage={"prompt_tokens": 0, "completion_tokens": 0},
                cached=False,
            )

        # Record the function call
        self.function_calls.append(function_call)

        # Return a result indicating a function call
        return CreateResult(
            finish_reason="function_calls",
            content=[function_call],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
            cached=False,
        )

    def set_expected_function_calls(self, function_calls: List[FunctionCall]):
        # Set the expected function calls for the test
        self.expected_calls = function_calls.copy()

    def get_function_calls(self) -> List[FunctionCall]:
        # Get the recorded function calls
        return self.function_calls

    def set_response(self, response: str):
        # Set the response to return after function execution
        self.responses.append(response)

    def actual_usage(self):
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def total_usage(self):
        return self.actual_usage()

    def count_tokens(self, messages, tools=[]):
        return 0

    def remaining_tokens(self, messages, tools=[]):
        return 1000

    @property
    def capabilities(self):
        return {
            "vision": False,
            "function_calling": True,
            "json_output": True,
        }

    # =============================================================================
    # test_utils/assertions.py
    # =============================================================================

    # Utility functions for asserting expected behaviors.

    def assert_function_call(
        function_calls: List[FunctionCall],
        expected_name: str,
        expected_arguments: dict,
    ):
        assert function_calls, "No function calls were made."
        function_call = function_calls[-1]
        assert (
            function_call.name == expected_name
        ), f"Expected function name '{expected_name}', got '{function_call.name}'"
        arguments = json.loads(function_call.arguments)
        assert arguments == expected_arguments, f"Expected arguments {expected_arguments}, got {arguments}"

    # =============================================================================
    # tools/math_tools.py
    # =============================================================================

    # Define some tools for testing.

    from pydantic import BaseModel

    class AddNumbersInput(BaseModel):
        x: float
        y: float

    class AddNumbersResult(BaseModel):
        result: float

    from autogen_core.tools import BaseTool

    class AddNumbersTool(BaseTool[AddNumbersInput, AddNumbersResult]):
        def __init__(self):
            super().__init__(
                args_type=AddNumbersInput,
                return_type=AddNumbersResult,
                name="add_numbers",
                description="Adds two numbers together.",
            )

        async def run(self, args: AddNumbersInput, cancellation_token: CancellationToken) -> AddNumbersResult:
            return AddNumbersResult(result=args.x + args.y)

    # =============================================================================
    # agents/calculator_agent.py
    # =============================================================================

    # An agent that uses a tool to perform calculations.

    from autogen_core import ToolAgent

    async def register_calculator_agent(runtime):
        tool = AddNumbersTool()
        tool_agent = ToolAgent(
            description="An agent that can perform mathematical operations.",
            tools=[tool],
        )
        await ToolAgent.register(
            runtime=runtime,
            type="CalculatorAgent",
            factory=lambda: tool_agent,
        )

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

    # =============================================================================
    # Implementation Details and Recommendations
    # =============================================================================

    # 1. **Mocking the ChatCompletionClient**

    #    - The `MockChatCompletionClient` simulates the behavior of the OpenAI LLM.
    #    - It allows you to predefine expected function calls and capture the function calls made by the agent.
    #    - This enables you to test whether the agent correctly interprets and generates function calls based on messages.

    # 2. **Intercepting and Verifying Function Calls**

    #    - The `get_function_calls()` method provides access to the list of function calls made during the test.
    #    - `assert_function_call()` checks that the function was called with the correct name and arguments.
    #    - This allows you to assert that the agent's logic for generating function calls is working as expected.

    # 3. **Testing Tools and Agents Separately**

    #    - Tools can be tested independently by directly invoking their `run` method with test inputs.
    #    - Agents can be tested by simulating messages and observing their interactions through the runtime.

    # 4. **Avoiding Real API Calls**

    #    - By using a mock client, you prevent actual API calls to OpenAI during testing.
    #    - This reduces costs and allows for consistent, repeatable tests.

    # 5. **Extensibility**

    #    - The mock client can be extended to simulate more complex behaviors, such as generating text outputs or handling errors.
    #    - Additional utility functions can be created for common assertions and test setups.

    # =============================================================================
    # Example Extension: Testing Text Output
    # =============================================================================

    # If you want to test text output from the LLM instead of function calls, you can modify the mock client.

    # In `MockChatCompletionClient`, add functionality to return predefined text responses.

    class MockChatCompletionClient(ChatCompletionClient):
        # ... existing methods ...

        def set_text_response(self, text: str):
            self.responses.append(text)

        async def create(
            self,
            # ... parameters ...
        ) -> CreateResult:
            # ... existing code ...
            if self.responses:
                text_response = self.responses.pop(0)
                return CreateResult(
                    finish_reason="stop",
                    content=text_response,
                    usage={"prompt_tokens": 0, "completion_tokens": 0},
                    cached=False,
                )
            # ... existing code ...

    # Then, in your test, you can set the expected text response and assert it.

    async def test_agent_text_response():
        # ... setup code ...

        chat_client.set_text_response("The result is 7.")

        # Run the agent interaction

        # ... code to invoke the agent ...

        # Assert the response
        assistant_message = generated_messages[-1]
        assert assistant_message.content == "The result is 7.", "Unexpected assistant response."


# =============================================================================
# Conclusion
# =============================================================================

# By creating a mock implementation of `ChatCompletionClient`, you can simulate the LLM's behavior and intercept
# function calls and text outputs for testing purposes. This approach allows you to validate that your agents and tools
# are interacting correctly, ensuring stability and correctness in a production environment.

# Key benefits include:
# - Testing without incurring API costs.
# - Control over the responses and behaviors of the LLM.
# - Ability to verify function names, parameters, and outputs.
# - Improved confidence in the system's reliability before deployment.

# Remember to integrate your tests into your CI/CD pipeline to catch issues early and maintain high code quality.

# **Explanation and Recommendations:**
#
# - **MockChatCompletionClient**: This custom client subclasses `ChatCompletionClient` and provides methods to intercept and verify function calls. It simulates the LLM's responses, either returning predefined function calls or text outputs.
#
# - **Interception and Verification**: By recording function calls and allowing you to set expected calls and responses, you can assert that the agent generates the correct function calls with the expected parameters.
#
# - **Testing Tools and Agents**: The test case demonstrates how to test an agent and its interaction with tools. This ensures that agents correctly interpret user messages, invoke tools as expected, and handle responses.
#
# - **Extensibility**: The testing framework can be extended to handle more complex scenarios, including error handling, multiple function calls, and varying responses.
#
# - **Avoiding Real API Calls**: Using a mock client prevents unnecessary API calls during testing, saving costs and allowing for consistent, controlled testing environments.
#
# - **Example Extension**: The testing framework can be adapted to verify text outputs from the LLM by adding methods to `MockChatCompletionClient`.
#
# **Best Practices:**
#
# - **Isolate Components**: Test agents, tools, and the runtime separately to pinpoint issues.
#
# - **Use Mocks and Stubs**: Replace external dependencies with mocks to create predictable and repeatable tests.
#
# - **Assert Expected Behaviors**: Use assertions to verify that the system behaves as intended in various scenarios.
#
# - **Automate Testing**: Integrate your tests into an automated testing framework, such as `pytest`, and include them in your CI/CD pipeline.
#
# - **Document Test Cases**: Provide clear documentation for each test case, explaining its purpose and expected outcomes.
#
# **Next Steps:**
#
# - **Expand Tests**: Create additional test cases covering edge cases, error conditions, and complex interactions.
#
# - **Parameterize Tests**: Use test parameterization to run the same test logic with different inputs and expected outputs.
#
# - **Logging and Diagnostics**: Incorporate logging within your tests to aid in debugging and traceability.
#
# - **Continuous Integration**: Configure your CI/CD pipeline to run the tests on each commit, ensuring code changes do not introduce regressions.
