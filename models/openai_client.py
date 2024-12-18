import openai
from autogen_core.models import ChatCompletionClient, CreateResult, LLMMessage, RequestUsage
from autogen_core import CancellationToken
from typing import List, Mapping, Any
from autogen_core.models import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    FunctionExecutionResultMessage,
    FunctionCall,
)
from autogen_core.tools import ToolSchema, Tool
import asyncio


class OpenAIChatCompletionClient(ChatCompletionClient):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        openai.api_key = api_key

    async def create(
        self,
        messages: List[LLMMessage],
        tools: List[Tool | ToolSchema] = [],
        json_output: bool = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken = None,
    ) -> CreateResult:
        # Convert messages to OpenAI API format
        api_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                api_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, UserMessage):
                api_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AssistantMessage):
                if isinstance(msg.content, str):
                    api_messages.append({"role": "assistant", "content": msg.content})
                else:
                    # Handle FunctionCall instances
                    for func_call in msg.content:
                        api_messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "function_call": {
                                    "name": func_call.name,
                                    "arguments": func_call.arguments,
                                },
                            }
                        )
            elif isinstance(msg, FunctionExecutionResultMessage):
                for result in msg.content:
                    api_messages.append(
                        {
                            "role": "function",
                            "name": result.call_id,
                            "content": result.content,
                        }
                    )

        # Prepare functions for the API
        api_functions = [tool.schema for tool in tools]

        # Call the OpenAI API
        response = await openai.ChatCompletion.acreate(
            model=self.model_name,
            messages=api_messages,
            functions=api_functions,
            function_call="auto",
            **extra_create_args,
        )

        # Extract the response
        choice = response.choices[0]
        message = choice.message
        content = message.get("content", None)

        if "function_call" in message:
            function_call = FunctionCall(
                id="unique_call_id",  # Generate or assign a unique ID
                name=message["function_call"]["name"],
                arguments=message["function_call"]["arguments"],
            )
            content = [function_call]

        return CreateResult(
            finish_reason=choice.finish_reason,
            content=content,
            usage=RequestUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            ),
            cached=False,
        )
