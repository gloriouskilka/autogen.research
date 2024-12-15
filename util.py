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
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from tracing import LangFuseExporter


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
print("LANGFUSE_PUBLIC_KEY:", os.getenv("LANGFUSE_PUBLIC_KEY"))
# print("LANGFUSE_SECRET_KEY:", os.getenv("LANGFUSE_SECRET_KEY"))
print("LANGFUSE_HOST:", os.getenv("LANGFUSE_HOST"))

# I want to intercept "create" function calls to model_clint to test the result later


class OpenAIChatCompletionClientWrapper(OpenAIChatCompletionClient):
    class FunctionCallVerification(Exception):
        result: CreateResult
        name: str
        arguments: dict

        def __init__(self, result: CreateResult, name: str, arguments: dict):
            self.result: CreateResult = result
            self.name: str = name
            self.arguments: dict = arguments

    def __init__(self, throw_on_create=False, expect_function_call=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.throw_on_create = throw_on_create
        self.expect_function_call = expect_function_call
        self.create_results: list[CreateResult] = []

    def set_throw_on_create(self, throw_on_create):
        self.throw_on_create = throw_on_create

    async def create(self, *args, **kwargs):
        result_coroutine = super().create(*args, **kwargs)

        # res = await result_coroutine
        # return res

        # Define a wrapper coroutine
        async def wrapper():
            result = await result_coroutine
            self.create_results.append(result)

            logger.debug(f"Intercepted create call: {result}")
            assert isinstance(result, CreateResult)

            if self.expect_function_call:
                assert result.finish_reason == "function_calls"
                assert isinstance(result.content, list)
                assert len(result.content) == 1
                function_call = result.content[0]
                assert isinstance(function_call, FunctionCall)
                arguments = json.loads(function_call.arguments)

                if self.throw_on_create:
                    raise self.FunctionCallVerification(result, function_call.name, arguments)
            else:
                if self.throw_on_create:
                    raise Exception("Not implemented")
            return result

        # Return the wrapper coroutine
        return await wrapper()


model_client = OpenAIChatCompletionClientWrapper(
    model="gpt-4o-mini",
    api_key=settings.openai_api_key,
)


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
