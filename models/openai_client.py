import asyncio
import json
import warnings
from asyncio import Task
from typing import (
    Optional,
    Dict,
    Any,
    Sequence,
    Mapping,
    Type,
    Union,
    cast,
    List,
    overload,
    TypeVar,
)

import httpx
from autogen_core import CancellationToken, Image, FunctionCall
from autogen_core.logging import LLMCallEvent
from autogen_core.models import (
    ModelCapabilities,
    LLMMessage,
    CreateResult,
    UserMessage,
    RequestUsage,
    ChatCompletionTokenLogprob,
    TopLogprob,
)
from autogen_core.tools import Tool, ToolSchema
from autogen_ext.models.openai import (
    OpenAIChatCompletionClient,
    OpenAIClientConfiguration,
)
from autogen_ext.models.openai._openai_client import (
    _openai_client_from_config,
    _create_args_from_config,
    create_kwargs,
    to_oai_type,
    normalize_name,
    _add_usage,
    assert_valid_name,
)
from loguru import logger
from openai import AsyncStream, APIConnectionError, APITimeoutError
from openai._base_client import get_platform
from openai._compat import model_copy
from openai._models import FinalRequestOptions
from openai._types import ResponseT, HttpxSendArgs
from openai._utils import asyncify
from openai.types import FunctionDefinition, FunctionParameters
from openai.types.chat import (
    ParsedChatCompletion,
    ChatCompletion,
    ParsedChoice,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel
from typing_extensions import Unpack


_AsyncStreamT = TypeVar("_AsyncStreamT", bound=AsyncStream[Any])


# Reference: https://platform.openai.com/docs/guides/function-calling
# https://json-schema.org/understanding-json-schema/
# Structured Outputs
# In August 2024, we launched Structured Outputs, which ensures that a model's output exactly matches a specified JSON schema.
#
# By default, when using function calling, the API will offer best-effort matching for your parameters, which means that occasionally the model may miss parameters or get their types wrong when using complicated schemas.
#
# You can enable Structured Outputs for function calling by setting the parameter strict: true in your function definition.
# You should also include the parameter additionalProperties: false and mark all arguments as required in your request. When this is enabled, the function arguments generated by the model will be constrained to match the JSON Schema provided in the function definition.
#
# As an alternative to function calling you can instead constrain the model's regular output to match a JSON Schema of your choosing. Learn more about when to use function calling vs when to control the model's normal output by using response_format.
#
# Parallel function calling and Structured Outputs
# When the model outputs multiple function calls via parallel function calling, model outputs may not match strict schemas supplied in tools.
#
# In order to ensure strict schema adherence, disable parallel function calls by supplying parallel_tool_calls: false. With this setting, the model will generate one function call at a time.

# Reference: https://platform.openai.com/docs/guides/structured-outputs#supported-schemas
# The following types are supported for Structured Outputs: String, Number, Boolean, Integer, Object, Array, Enum, anyOf


def convert_tools(
    tools: Sequence[Tool | ToolSchema],
) -> List[ChatCompletionToolParam]:
    result: List[ChatCompletionToolParam] = []
    for tool in tools:
        if isinstance(tool, Tool):
            tool_schema = tool.schema
        else:
            assert isinstance(tool, dict)
            tool_schema = tool

        param_dict: dict = ChatCompletionToolParam(
            type="function",
            function=FunctionDefinition(
                name=tool_schema["name"],
                description=tool_schema.get("description", ""),
                parameters=cast(FunctionParameters, tool_schema.get("parameters", {})),
                strict=True,
            ).model_dump(),
        )
        parameters_dict = param_dict["function"]["parameters"]
        parameters_dict["additionalProperties"] = False

        result.append(param_dict)
        assert_valid_name(param_dict["function"]["name"])

    # result = [
    #     {
    #         "function": {
    #             "description": "DAVAI",
    #             "name": "decide_filters",
    #             "parameters": {
    #                 "additionalProperties": False,
    #                 "properties": {
    #                     "filters_mapped": {
    #                         "properties": {
    #                             "filters": {
    #                                 "anyOf": [
    #                                     {
    #                                         "additionalProperties": False,  # TODO: Modified manually, but also removed | None from Filters
    #                                         "type": "object",
    #                                     },
    #                                     {"type": "null"},
    #                                 ],
    #                                 "title": "Filters",
    #                             },
    #                             "reason": {"title": "Reason", "type": "string"},
    #                             "successful": {"title": "Successful", "type": "boolean"},
    #                         },
    #                         "required": ["reason", "filters", "successful"],
    #                         "title": "Filters",
    #                         "type": "object",
    #                         "additionalProperties": False,  # TODO: ADDED MANUALLY
    #                     }
    #                 },
    #                 "required": ["filters_mapped"],
    #                 "type": "object",
    #             },
    #             "strict": True,
    #         },
    #         "type": "function",
    #     }
    # ]

    return result


class OpenAIChatCompletionClientStructuredOutput(OpenAIChatCompletionClient):
    """Chat completion client for OpenAI hosted models.

    You can also use this client for OpenAI-compatible ChatCompletion endpoints.
    **Using this client for non-OpenAI models is not tested or guaranteed.**

    For non-OpenAI models, please first take a look at our `community extensions <https://microsoft.github.io/autogen/dev/user-guide/extensions-user-guide/index.html>`_
    for additional model clients.

    Args:
        model (str): The model to use. **Required.**
        api_key (str): The API key to use. **Required if 'OPENAI_API_KEY' is not found in the environment variables.**
        timeout (optional, int): The timeout for the request in seconds.
        max_retries (optional, int): The maximum number of retries to attempt.
        organization_id (optional, str): The organization ID to use.
        base_url (optional, str): The base URL to use. **Required if the model is not hosted on OpenAI.**
        model_capabilities (optional, ModelCapabilities): The capabilities of the model. **Required if the model name is not a valid OpenAI model.**

    To use this client, you must install the `openai` extension:

        .. code-block:: bash

            pip install 'autogen-ext[openai]==0.4.0.dev8'

    The following code snippet shows how to use the client with an OpenAI model:

        .. code-block:: python

            from autogen_ext.models.openai import OpenAIChatCompletionClient
            from autogen_core.models import UserMessage

            openai_client = OpenAIChatCompletionClient(
                model="gpt-4o-2024-08-06",
                # api_key="sk-...", # Optional if you have an OPENAI_API_KEY environment variable set.
            )

            result = await openai_client.create([UserMessage(content="What is the capital of France?", source="user")])  # type: ignore
            print(result)


    To use the client with a non-OpenAI model, you need to provide the base URL of the model and the model capabilities:

        .. code-block:: python

            from autogen_ext.models.openai import OpenAIChatCompletionClient

            custom_model_client = OpenAIChatCompletionClient(
                model="custom-model-name",
                base_url="https://custom-model.com/reset/of/the/path",
                api_key="placeholder",
                model_capabilities={
                    "vision": True,
                    "function_calling": True,
                    "json_output": True,
                },
            )

    """

    def __init__(self, **kwargs: Unpack[OpenAIClientConfiguration]):
        # if "model" not in kwargs:
        #     raise ValueError("model is required for OpenAIChatCompletionClient")
        #
        # model_capabilities: Optional[ModelCapabilities] = None
        # copied_args = dict(kwargs).copy()
        # if "model_capabilities" in kwargs:
        #     model_capabilities = kwargs["model_capabilities"]
        #     del copied_args["model_capabilities"]
        #
        # client = _openai_client_from_config(copied_args)
        # create_args = _create_args_from_config(copied_args)
        # self._raw_config = copied_args
        super().__init__(**kwargs)

    # def __getstate__(self) -> Dict[str, Any]:
    #     state = self.__dict__.copy()
    #     state["_client"] = None
    #     return state
    #
    # def __setstate__(self, state: Dict[str, Any]) -> None:
    #     self.__dict__.update(state)
    #     self._client = _openai_client_from_config(state["_raw_config"])

    async def create(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        # Make sure all extra_create_args are valid
        extra_create_args_keys = set(extra_create_args.keys())
        if not create_kwargs.issuperset(extra_create_args_keys):
            raise ValueError(
                f"Extra create args are invalid: {extra_create_args_keys - create_kwargs}"
            )

        # Copy the create args and overwrite anything in extra_create_args
        create_args = self._create_args.copy()
        create_args.update(extra_create_args)

        # Declare use_beta_client
        use_beta_client: bool = False
        response_format_value: Optional[Type[BaseModel]] = None

        if "response_format" in create_args:
            value = create_args["response_format"]
            # If value is a Pydantic model class, use the beta client
            if isinstance(value, type) and issubclass(value, BaseModel):
                response_format_value = value
                use_beta_client = True
            else:
                # response_format_value is not a Pydantic model class
                use_beta_client = False
                response_format_value = None

        # Remove 'response_format' from create_args to prevent passing it twice
        create_args_no_response_format = {
            k: v for k, v in create_args.items() if k != "response_format"
        }

        # TODO: allow custom handling.
        # For now we raise an error if images are present and vision is not supported
        if self.capabilities["vision"] is False:
            for message in messages:
                if isinstance(message, UserMessage):
                    if isinstance(message.content, list) and any(
                        isinstance(x, Image) for x in message.content
                    ):
                        raise ValueError(
                            "Model does not support vision and image was provided"
                        )

        if json_output is not None:
            if self.capabilities["json_output"] is False and json_output is True:
                raise ValueError("Model does not support JSON output")

            if json_output is True:
                create_args["response_format"] = {"type": "json_object"}
            else:
                create_args["response_format"] = {"type": "text"}

        if self.capabilities["json_output"] is False and json_output is True:
            raise ValueError("Model does not support JSON output")

        oai_messages_nested = [to_oai_type(m) for m in messages]
        oai_messages = [item for sublist in oai_messages_nested for item in sublist]

        if self.capabilities["function_calling"] is False and len(tools) > 0:
            raise ValueError("Model does not support function calling")
        future: Union[Task[ParsedChatCompletion[BaseModel]], Task[ChatCompletion]]
        if len(tools) > 0:
            converted_tools = convert_tools(tools)
            if use_beta_client:
                # Pass response_format_value if it's not None
                if response_format_value is not None:
                    future = asyncio.ensure_future(
                        self._client.beta.chat.completions.parse(
                            messages=oai_messages,
                            tools=converted_tools,
                            response_format=response_format_value,
                            **create_args_no_response_format,
                        )
                    )
                else:
                    future = asyncio.ensure_future(
                        self._client.beta.chat.completions.parse(
                            messages=oai_messages,
                            tools=converted_tools,
                            **create_args_no_response_format,
                        )
                    )
            else:
                future = asyncio.ensure_future(
                    self._client.chat.completions.create(
                        messages=oai_messages,
                        stream=False,
                        tools=converted_tools,
                        **create_args,
                    )
                )
        else:
            if use_beta_client:
                if response_format_value is not None:
                    future = asyncio.ensure_future(
                        self._client.beta.chat.completions.parse(
                            messages=oai_messages,
                            response_format=response_format_value,
                            **create_args_no_response_format,
                        )
                    )
                else:
                    future = asyncio.ensure_future(
                        self._client.beta.chat.completions.parse(
                            messages=oai_messages,
                            **create_args_no_response_format,
                        )
                    )
            else:
                future = asyncio.ensure_future(
                    self._client.chat.completions.create(
                        messages=oai_messages,
                        stream=False,
                        **create_args,
                    )
                )

        if cancellation_token is not None:
            cancellation_token.link_future(future)
        result: Union[ParsedChatCompletion[BaseModel], ChatCompletion] = await future
        if use_beta_client:
            result = cast(ParsedChatCompletion[Any], result)

        if result.usage is not None:
            logger.info(
                LLMCallEvent(
                    prompt_tokens=result.usage.prompt_tokens,
                    completion_tokens=result.usage.completion_tokens,
                )
            )

        usage = RequestUsage(
            # TODO backup token counting
            prompt_tokens=result.usage.prompt_tokens if result.usage is not None else 0,
            completion_tokens=(
                result.usage.completion_tokens if result.usage is not None else 0
            ),
        )

        if self._resolved_model is not None:
            if self._resolved_model != result.model:
                warnings.warn(
                    f"Resolved model mismatch: {self._resolved_model} != {result.model}. Model mapping may be incorrect.",
                    stacklevel=2,
                )

        # Limited to a single choice currently.
        choice: Union[ParsedChoice[Any], ParsedChoice[BaseModel], Choice] = (
            result.choices[0]
        )
        if choice.finish_reason == "function_call":
            raise ValueError("Function calls are not supported in this context")

        content: Union[str, List[FunctionCall]]
        if choice.finish_reason == "tool_calls":
            assert choice.message.tool_calls is not None
            assert choice.message.function_call is None

            # NOTE: If OAI response type changes, this will need to be updated
            content = [
                FunctionCall(
                    id=x.id,
                    arguments=x.function.arguments,
                    name=normalize_name(x.function.name),
                )
                for x in choice.message.tool_calls
            ]
            finish_reason = "function_calls"
        else:
            finish_reason = choice.finish_reason
            content = choice.message.content or ""
        logprobs: Optional[List[ChatCompletionTokenLogprob]] = None
        if choice.logprobs and choice.logprobs.content:
            logprobs = [
                ChatCompletionTokenLogprob(
                    token=x.token,
                    logprob=x.logprob,
                    top_logprobs=[
                        TopLogprob(logprob=y.logprob, bytes=y.bytes)
                        for y in x.top_logprobs
                    ],
                    bytes=x.bytes,
                )
                for x in choice.logprobs.content
            ]
        response = CreateResult(
            finish_reason=finish_reason,  # type: ignore
            content=content,
            usage=usage,
            cached=False,
            logprobs=logprobs,
        )

        _add_usage(self._actual_usage, usage)
        _add_usage(self._total_usage, usage)

        # TODO - why is this cast needed?
        return response

    #
    # @overload
    # async def request(
    #     self,
    #     cast_to: Type[ResponseT],
    #     options: FinalRequestOptions,
    #     *,
    #     stream: bool,
    #     stream_cls: type[_AsyncStreamT] | None = None,
    #     remaining_retries: Optional[int] = None,
    # ) -> ResponseT | _AsyncStreamT: ...
    #
    # async def request(
    #     self,
    #     cast_to: Type[ResponseT],
    #     options: FinalRequestOptions,
    #     *,
    #     stream: bool = False,
    #     stream_cls: type[_AsyncStreamT] | None = None,
    #     remaining_retries: Optional[int] = None,
    # ) -> ResponseT | _AsyncStreamT:
    #     if remaining_retries is not None:
    #         retries_taken = options.get_max_retries(self.max_retries) - remaining_retries
    #     else:
    #         retries_taken = 0
    #
    #     return await self._request(
    #         cast_to=cast_to,
    #         options=options,
    #         stream=stream,
    #         stream_cls=stream_cls,
    #         retries_taken=retries_taken,
    #     )
    #
    # async def _request(
    #     self,
    #     cast_to: Type[ResponseT],
    #     options: FinalRequestOptions,
    #     *,
    #     stream: bool,
    #     stream_cls: type[_AsyncStreamT] | None,
    #     retries_taken: int,
    # ) -> ResponseT | _AsyncStreamT:
    #     if self._platform is None:
    #         # `get_platform` can make blocking IO calls so we
    #         # execute it earlier while we are in an async context
    #         self._platform = await asyncify(get_platform)()
    #
    #     # create a copy of the options we were given so that if the
    #     # options are mutated later & we then retry, the retries are
    #     # given the original options
    #     input_options = model_copy(options)
    #
    #     cast_to = self._maybe_override_cast_to(cast_to, options)
    #     options = await self._prepare_options(options)
    #
    #     remaining_retries = options.get_max_retries(self.max_retries) - retries_taken
    #     request = self._build_request(options, retries_taken=retries_taken)
    #     await self._prepare_request(request)
    #
    #     kwargs: HttpxSendArgs = {}
    #     if self.custom_auth is not None:
    #         kwargs["auth"] = self.custom_auth
    #
    #     try:
    #         response = await self._client.send(
    #             request,
    #             stream=stream or self._should_stream_response_body(request=request),
    #             **kwargs,
    #         )
    #     except httpx.TimeoutException as err:
    #         logger.debug("Encountered httpx.TimeoutException", exc_info=True)
    #
    #         if remaining_retries > 0:
    #             return await self._retry_request(
    #                 input_options,
    #                 cast_to,
    #                 retries_taken=retries_taken,
    #                 stream=stream,
    #                 stream_cls=stream_cls,
    #                 response_headers=None,
    #             )
    #
    #         logger.debug("Raising timeout error")
    #         raise APITimeoutError(request=request) from err
    #     except Exception as err:
    #         logger.debug("Encountered Exception", exc_info=True)
    #
    #         if remaining_retries > 0:
    #             return await self._retry_request(
    #                 input_options,
    #                 cast_to,
    #                 retries_taken=retries_taken,
    #                 stream=stream,
    #                 stream_cls=stream_cls,
    #                 response_headers=None,
    #             )
    #
    #         logger.debug("Raising connection error")
    #         raise APIConnectionError(request=request) from err
    #
    #     logger.debug(
    #         'HTTP Request: %s %s "%i %s"', request.method, request.url, response.status_code, response.reason_phrase
    #     )
    #
    #     try:
    #         response.raise_for_status()
    #     except httpx.HTTPStatusError as err:  # thrown on 4xx and 5xx status code
    #         logger.debug("Encountered httpx.HTTPStatusError", exc_info=True)
    #
    #         if remaining_retries > 0 and self._should_retry(err.response):
    #             await err.response.aclose()
    #             return await self._retry_request(
    #                 input_options,
    #                 cast_to,
    #                 retries_taken=retries_taken,
    #                 response_headers=err.response.headers,
    #                 stream=stream,
    #                 stream_cls=stream_cls,
    #             )
    #
    #         # If the response is streamed then we need to explicitly read the response
    #         # to completion before attempting to access the response text.
    #         if not err.response.is_closed:
    #             await err.response.aread()
    #
    #         logger.debug("Re-raising status error")
    #         raise self._make_status_error_from_response(err.response) from None
    #
    #     return await self._process_response(
    #         cast_to=cast_to,
    #         options=options,
    #         response=response,
    #         stream=stream,
    #         stream_cls=stream_cls,
    #         retries_taken=retries_taken,
    #     )


class Verification(Exception):
    result: CreateResult

    def __init__(self, result: CreateResult):
        self.result = result


class FunctionCallRecord(BaseModel):
    function_name: str
    arguments: dict
    # You can include other fields as necessary, such as the function call id, etc.


class FunctionCallVerification(Verification):
    function_calls: list[FunctionCallRecord]

    def __init__(
        self,
        result: CreateResult,
        function_calls: list[FunctionCallRecord],
    ):
        super().__init__(result)
        self.function_calls = function_calls


class TextResultVerification(Verification):
    content: str

    def __init__(self, result: CreateResult):
        super().__init__(result)
        self.content = result.content


class OpenAIChatCompletionClientStructuredOutputWithCreateIntercept(
    OpenAIChatCompletionClientStructuredOutput
):
    def __init__(self, throw_on_create=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.throw_on_create = throw_on_create
        self.create_results: list[Verification] = []

    def set_throw_on_create(self, throw_on_create):
        self.throw_on_create = throw_on_create

    async def create(self, *args, **kwargs):
        result_coroutine = super().create(*args, **kwargs)

        # Define a wrapper coroutine
        async def wrapper():
            result = await result_coroutine

            logger.debug(f"Intercepted create call: {result}")
            assert isinstance(result, CreateResult)

            verification = None  # Initialize verification object

            if result.finish_reason == "function_calls":
                # Handle function calls
                assert isinstance(
                    result.content, list
                ), "Expected result.content to be a list"
                if len(result.content) == 0:
                    raise Exception("No function calls returned.")

                function_calls = []
                for function_call in result.content:
                    assert isinstance(
                        function_call, FunctionCall
                    ), f"Expected FunctionCall, got {type(function_call)}"
                    function_call_record = FunctionCallRecord(
                        function_name=function_call.name,
                        arguments=json.loads(function_call.arguments),
                    )
                    function_calls.append(function_call_record)

                # After collecting all function calls, create the verification object
                verification = FunctionCallVerification(result, function_calls)

            elif result.finish_reason == "stop":
                # Handle text results
                assert isinstance(
                    result.content, str
                ), "Expected result.content to be a string"

                verification = TextResultVerification(result)

            else:
                raise Exception(f"Unexpected finish_reason: {result.finish_reason}")

            if self.throw_on_create:
                raise verification
            else:
                self.create_results.append(verification)

            return result

        # Return the wrapper coroutine
        return await wrapper()


# Almost copy-paste, but inherited from different class


class OpenAIChatCompletionClientWithCreateIntercept(OpenAIChatCompletionClient):
    def __init__(self, throw_on_create=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.throw_on_create = throw_on_create
        self.create_results: list[Verification] = []

    def set_throw_on_create(self, throw_on_create):
        self.throw_on_create = throw_on_create

    async def create(self, *args, **kwargs):
        result_coroutine = super().create(*args, **kwargs)

        # Define a wrapper coroutine
        async def wrapper():
            result = await result_coroutine

            logger.debug(f"Intercepted create call: {result}")
            assert isinstance(result, CreateResult)

            verification = None  # Initialize verification object

            if result.finish_reason == "function_calls":
                # Handle function calls
                assert isinstance(
                    result.content, list
                ), "Expected result.content to be a list"
                if len(result.content) == 0:
                    raise Exception("No function calls returned.")

                function_calls = []
                for function_call in result.content:
                    assert isinstance(
                        function_call, FunctionCall
                    ), f"Expected FunctionCall, got {type(function_call)}"
                    function_call_record = FunctionCallRecord(
                        function_name=function_call.name,
                        arguments=json.loads(function_call.arguments),
                    )
                    function_calls.append(function_call_record)

                # After collecting all function calls, create the verification object
                verification = FunctionCallVerification(result, function_calls)

            elif result.finish_reason == "stop":
                # Handle text results
                assert isinstance(
                    result.content, str
                ), "Expected result.content to be a string"

                verification = TextResultVerification(result)

            else:
                raise Exception(f"Unexpected finish_reason: {result.finish_reason}")

            if self.throw_on_create:
                raise verification
            else:
                self.create_results.append(verification)

            return result

        # Return the wrapper coroutine
        return await wrapper()
