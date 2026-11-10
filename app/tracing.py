"""Optional OpenTelemetry tracing.

Metrics tell you "p95 is 300ms"; traces tell you "this particular 300ms was
40ms decoding, 250ms inference and 10ms encoding". The first finds problems,
the second locates them; neither replaces the other.

Off by default: enabled only when OTEL_EXPORTER_OTLP_ENDPOINT is set, which
docker-compose.yml does. When it is not set, opentelemetry-api hands out
non-recording spans and every call is a no-op, so the image can run on its
own with docker run at close to zero cost.

Only the API package (small, pure Python) is imported at module level; the
SDK, the exporter and the FastAPI auto-instrumentation are imported inside
setup(), so a slim install still starts.

Versions must match. Core 1.X.0 pairs with instrumentation 0.(X+21)b0; this
project uses 1.27.0 <-> 0.48b0. Mixed versions typically show up as an
ImportError, or, harder to debug, as spans silently never being exported.
"""

import logging
from contextlib import contextmanager
from typing import Optional

from opentelemetry import trace

from app import config

logger = logging.getLogger("yolo-service")

# This is the *instrumentation scope*, not the service name — it says which
# code produced a span (visible as otel.library.name on each span), whereas
# service.name in setup() says which process it came from. Two different
# things, so they should not share a string.
_tracer = trace.get_tracer(__name__)
_enabled = False


def setup(app=None) -> bool:
    """Install the OTLP exporter. Returns whether tracing is actually enabled."""
    global _enabled

    endpoint = config.otlp_endpoint()
    if not endpoint:
        logger.info(
            '{"event": "tracing_disabled", '
            '"reason": "OTEL_EXPORTER_OTLP_ENDPOINT not set"}'
        )
        return False

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create({
                "service.name": config.service_name(),
                "service.version": config.service_version(),
            })
        )
        # insecure=True: the Collector sits on the same compose network, not
        # the internet, so there is no TLS to negotiate. Note that the gRPC
        # endpoint is host:port **without an http:// prefix**; adding one fails.
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
        )
        trace.set_tracer_provider(provider)

        if app is not None:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            # Prometheus scrapes /metrics every 10 seconds and the healthcheck
            # hits /health even more often. Those spans carry no value and
            # would bury the real requests.
            FastAPIInstrumentor.instrument_app(app, excluded_urls="health,metrics")
    except Exception as exc:  # broken telemetry is minor; a broken service is not
        logger.warning('{"event": "tracing_setup_failed", "error": "%s"}', exc)
        return False

    _enabled = True
    logger.info('{"event": "tracing_enabled", "endpoint": "%s"}', endpoint)
    return True


def enabled() -> bool:
    return _enabled


@contextmanager
def span(name: str, **attributes):
    """Open a child span. A no-op when tracing is off.

    Span attributes and metric labels follow **completely different**
    cardinality rules: image.width=1280 is fine here, because a span is a
    discrete event rather than an aggregated time series; the same value as a
    metric label would be a disaster.
    """
    with _tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield current


def annotate(**attributes) -> None:
    """Attach attributes to the currently active span."""
    current = trace.get_current_span()
    for key, value in attributes.items():
        if value is not None:
            current.set_attribute(key, value)


def current_trace_id() -> Optional[str]:
    """Trace id of the current span (32 hex characters).

    Written into the access log, it lets you jump from an error log line
    straight to the full trace in Jaeger (log-trace correlation).
    """
    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return None
    return format(context.trace_id, "032x")
