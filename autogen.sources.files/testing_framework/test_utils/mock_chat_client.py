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
