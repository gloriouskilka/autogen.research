import os

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


settings = Settings()

# to set env vars - used by langfuse
load_dotenv()

# Print those env vars:
print("LANGFUSE_PUBLIC_KEY:", os.getenv("LANGFUSE_PUBLIC_KEY"))
# print("LANGFUSE_SECRET_KEY:", os.getenv("LANGFUSE_SECRET_KEY"))
print("LANGFUSE_HOST:", os.getenv("LANGFUSE_HOST"))

model_client = OpenAIChatCompletionClient(
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
