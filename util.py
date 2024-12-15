import inspect
import os

from autogen_core.models import CreateResult
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from langfuse import Langfuse
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

    class CreateResultException(Exception):
        def __init__(self, result: CreateResult):
            self.result: CreateResult = result

    def __init__(self, throw_on_create=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.throw_on_create = throw_on_create
        # self.on_create = None
        # List of results to test later
        self.create_results = []

    # def set_on_create(self, on_create):
    #     self.on_create = on_create
    def set_throw_on_create(self, throw_on_create):
        self.throw_on_create = throw_on_create

    async def create(self, *args, **kwargs):
        result_original = super().create(*args, **kwargs)
        if inspect.isawaitable(result_original):
            result = await result_original
        else:
            result = result_original
        self.create_results.append(result)

        if self.throw_on_create:
            assert isinstance(result, CreateResult)
            raise self.CreateResultException(result)
        return result_original


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
