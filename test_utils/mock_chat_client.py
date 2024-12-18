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
        else:
            content = messages[-1].content if messages else ""
            logger.debug(f"No expected_calls left. Returning echo of user's message: {content}")
            return CreateResult(
                finish_reason="stop",
                content=content,
                usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
                cached=False,
            )

        self.function_calls.append(function_call)
        logger.debug(f"Appended function_call to function_calls. Current size: {len(self.function_calls)}")

        return CreateResult(
            finish_reason="function_calls",
            content=[function_call],
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

    # def actual_usage(self):
    #     usage = {
    #         "prompt_tokens": 0,
    #         "completion_tokens": 0,
    #     }
    #     logger.debug(f"Returning actual usage: {usage}")
    #     return usage
    #
    # def total_usage(self):
    #     total_usage = self.actual_usage()
    #     logger.debug(f"Returning total usage: {total_usage}")
    #     return total_usage
    #
    # def count_tokens(self, messages, tools=None):
    #     if tools is None:
    #         tools = []
    #         logger.debug("tools was None, initialized to empty list in count_tokens.")
    #     token_count = 0
    #     logger.debug(f"Returning token count: {token_count}")
    #     return token_count
    #
    # def remaining_tokens(self, messages, tools=None):
    #     if tools is None:
    #         tools = []
    #         logger.debug("tools was None, initialized to empty list in remaining_tokens.")
    #     remaining = 1000
    #     logger.debug(f"Returning remaining tokens: {remaining}")
    #     return remaining

    @property
    def capabilities(self):
        capabilities = {
            "vision": False,
            "function_calling": True,
            "json_output": True,
        }
        logger.debug(f"Returning capabilities: {capabilities}")
        return capabilities
