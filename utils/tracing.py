from langfuse import Langfuse
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, BatchSpanProcessor, SpanExportResult


class LangFuseExporter(SpanExporter):
    def __init__(self, langfuse_client: Langfuse):
        self.langfuse = langfuse_client

    def export(self, spans):
        for span in spans:
            # Extract data from the OpenTelemetry Span
            trace_id = span.get_span_context().trace_id
            span_id = span.get_span_context().span_id
            parent_span_id = span.parent.span_id if span.parent else None
            name = span.name
            start_time_ns = span.start_time
            end_time_ns = span.end_time
            attributes = span.attributes
            status = span.status

            # Convert trace_id and span_id to hex string
            trace_id_hex = trace_id_to_hex(trace_id)
            span_id_hex = span_id_to_hex(span_id)
            parent_span_id_hex = span_id_to_hex(parent_span_id) if parent_span_id else None

            # Convert times from nanoseconds to datetime objects
            start_time_dt = ns_to_datetime(start_time_ns)
            end_time_dt = ns_to_datetime(end_time_ns)

            # Prepare metadata or any other data as needed
            metadata = dict(attributes)

            # Send data to LangFuse using the Python API
            self.langfuse.span(
                trace_id=trace_id_hex,
                id=span_id_hex,
                parent_id=parent_span_id_hex,
                name=name,
                start_time=start_time_dt,
                end_time=end_time_dt,
                metadata=metadata,
                status_message=status.description if status else None,
                status_code=str(status.status_code) if status else None,
            )

        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000):
        pass


def trace_id_to_hex(trace_id):
    return format(trace_id, "032x")


def span_id_to_hex(span_id):
    return format(span_id, "016x") if span_id else None


def ns_to_datetime(timestamp_ns):
    from datetime import datetime, timezone

    timestamp_sec = timestamp_ns / 1e9  # Convert nanoseconds to seconds
    return datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)


def configure_tracing(langfuse_client: Langfuse):
    tracer_provider = TracerProvider(
        resource=Resource(
            attributes={
                "service.name": "my-service",
                "service.version": "1.0.0",
            }
        )
    )
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
