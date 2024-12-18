# =============================================================================
# test_utils/mock_chat_client.py
# =============================================================================

# A mock implementation of ChatCompletionClient that allows interception and validation
# of function calls.

from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    FunctionExecutionResultMessage,
    RequestUsage,
)
from autogen_core._types import FunctionCall
from autogen_core.models import AssistantMessage, FunctionExecutionResult
from autogen_core.tools import Tool, ToolSchema
from typing import List, Optional, Mapping, Any, Sequence
from autogen_core import CancellationToken
import json
from loguru import logger


class MockChatCompletionClient(ChatCompletionClient):
    def __init__(self):
        self.function_calls = []
        self.responses = []
        self.expected_calls = []
        logger.debug("Initialized MockChatCompletionClient with empty function_calls, responses, and expected_calls.")

    async def create(
        self,
        messages: Sequence[LLMMessage],
        tools=None,
        json_output: Optional[bool] = None,
        extra_create_args=None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        logger.debug("Entering create method.")
        if extra_create_args is None:
            extra_create_args = {}
            logger.debug("extra_create_args was None, initialized to empty dictionary.")
        if tools is None:
            tools = []
            logger.debug("tools was None, initialized to empty list.")

        logger.debug(f"Current expected_calls size: {len(self.expected_calls)}")
        if self.expected_calls:
            function_call = self.expected_calls.pop(0)
            logger.debug(f"Popped function_call from expected_calls: {function_call}")
            self.function_calls.append(function_call)
            logger.debug(f"Appended function_call to function_calls. Current size: {len(self.function_calls)}")
            # Return the function call request
            return CreateResult(
                finish_reason="function_calls",
                content=[function_call],
                usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
                cached=False,
            )
        else:
            # Generate a mock assistant response
            last_message = messages[-1] if messages else None
            if isinstance(last_message, FunctionExecutionResultMessage):
                # If the last message is a function execution result, the assistant should process it
                # For simplicity, let's simulate the assistant returning the 'result' from the function

                # Check if content is a list
                if isinstance(last_message.content, list):
                    # Extract the content from the list
                    # Assuming it's a list of FunctionExecutionResults
                    function_execution_result = last_message.content[0]
                    function_result_str = function_execution_result.content
                else:
                    function_result_str = last_message.content

                function_result = json.loads(function_result_str)
                content = str(function_result.get("result", ""))
                logger.debug(f"No expected_calls left. Returning processed function result: {content}")
            else:
                # Echo the user's message
                content = last_message.content if last_message else ""
                logger.debug(f"No expected_calls left. Returning echo of user's message: {content}")

            return CreateResult(
                finish_reason="stop",
                content=content,
                usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
                cached=False,
            )

    def set_expected_function_calls(self, function_calls: List[FunctionCall]):
        self.expected_calls = function_calls.copy()
        logger.debug(f"Set expected_calls with a copy of function_calls. Size: {len(self.expected_calls)}")

    def get_function_calls(self) -> List[FunctionCall]:
        logger.debug(f"Returning function_calls. Size: {len(self.function_calls)}")
        return self.function_calls

    def set_response(self, response: str):
        self.responses.append(response)
        logger.debug(f"Appended response to responses. Current size: {len(self.responses)}")

    @property
    def capabilities(self):
        capabilities = {
            "vision": False,
            "function_calling": True,
            "json_output": True,
        }
        logger.debug(f"Returning capabilities: {capabilities}")
        return capabilities
