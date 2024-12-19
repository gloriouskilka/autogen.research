import json

from autogen_core import FunctionCall
from autogen_core.models import CreateResult
from autogen_ext.models.openai import OpenAIChatCompletionClient
from loguru import logger
from pydantic import BaseModel


class OpenAIChatCompletionClientWrapper(OpenAIChatCompletionClient):
    class Verification(Exception):
        result: CreateResult

        def __init__(self, result: CreateResult):
            self.result = result

    class FunctionCallRecord(BaseModel):
        function_name: str
        arguments: dict
        # You can include other fields as necessary, such as the function call id, etc.

    class FunctionCallVerification(Verification):
        function_calls: list["OpenAIChatCompletionClientWrapper.FunctionCallRecord"]

        def __init__(
            self, result: CreateResult, function_calls: list["OpenAIChatCompletionClientWrapper.FunctionCallRecord"]
        ):
            super().__init__(result)
            self.function_calls = function_calls

    class TextResultVerification(Verification):
        content: str

        def __init__(self, result: CreateResult):
            super().__init__(result)
            self.content = result.content

    def __init__(self, throw_on_create=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.throw_on_create = throw_on_create
        self.create_results: list[OpenAIChatCompletionClientWrapper.Verification] = []

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
                assert isinstance(result.content, list), "Expected result.content to be a list"
                if len(result.content) == 0:
                    raise Exception("No function calls returned.")

                function_calls = []
                for function_call in result.content:
                    assert isinstance(function_call, FunctionCall), f"Expected FunctionCall, got {type(function_call)}"
                    function_call_record = self.FunctionCallRecord(
                        function_name=function_call.name, arguments=json.loads(function_call.arguments)
                    )
                    function_calls.append(function_call_record)

                # After collecting all function calls, create the verification object
                verification = self.FunctionCallVerification(result, function_calls)

            elif result.finish_reason == "stop":
                # Handle text results
                assert isinstance(result.content, str), "Expected result.content to be a string"

                verification = self.TextResultVerification(result)

            else:
                raise Exception(f"Unexpected finish_reason: {result.finish_reason}")

            if self.throw_on_create:
                raise verification
            else:
                self.create_results.append(verification)

            return result

        # Return the wrapper coroutine
        return await wrapper()
