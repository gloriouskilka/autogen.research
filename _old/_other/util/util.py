import inspect
import json
import os

from autogen_core import FunctionCall
from autogen_core.models import CreateResult
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from langfuse import Langfuse
from loguru import logger
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from utils.tracing import LangFuseExporter


# from util.tracing import LangFuseExporter


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

    openai_api_key: str = Field()

    langfuse_secret_key: str = Field(default=None)
    langfuse_public_key: str = Field(default=None)
    langfuse_host: str = Field(default=None)

    weatherstack_api_key: str = Field(default=None)


settings = Settings()

# to set env vars - used by langfuse
load_dotenv()

# Print those env vars:
logger.debug("LANGFUSE_PUBLIC_KEY:", os.getenv("LANGFUSE_PUBLIC_KEY"))
# logger.debug("LANGFUSE_SECRET_KEY:", os.getenv("LANGFUSE_SECRET_KEY"))
logger.debug("LANGFUSE_HOST:", os.getenv("LANGFUSE_HOST"))

# I want to intercept "create" function calls to model_clint to test the result later


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


class QNA(BaseModel):
    query: str
    name: str
    arguments: dict


def configure_tracing(langfuse_client: Langfuse):
    resource = Resource(
        attributes={
            "service.name": "my-service",
            "service.version": "1.0.0",
        }
    )

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(LangFuseExporter(langfuse_client=langfuse_client)))

    # # Adding Graphana / Prometheus
    # tracer_provider.add_span_processor(
    #     BatchSpanProcessor(
    #         OTLPSpanExporter(
    #             endpoint="http://localhost:4317",
    #         )
    #     )
    # )

    trace.set_tracer_provider(tracer_provider)
    return tracer_provider


# def configure_oltp_tracing(endpoint: str = None) -> trace.TracerProvider:
#     # Configure Tracing
#     tracer_provider = TracerProvider(resource=Resource({"service.name": "my-service"}))
#     processor = BatchSpanProcessor(OTLPSpanExporter())
#     tracer_provider.add_span_processor(processor)
#     trace.set_tracer_provider(tracer_provider)
#
#     return tracer_provider
