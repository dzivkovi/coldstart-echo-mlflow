"""Lightweight, standard-library request timing for a Databricks pyfunc or service.

Zero third-party dependencies (only logging, time, contextlib) - safe to use where
external libraries are restricted. It captures DURATIONS ONLY, never request/response
payloads, so there is nothing sensitive to govern. It emits one summary log line per
request with each phase's duration and its PERCENTAGE of the total, so time can be
optimised in priority order rather than by guessing.

For the why, the two-layer (outside/inside) model, the MLflow Tracing alternative,
and the data-governance trade-off, see the performance-monitoring doc in the
companion `genai-coldstart-guard` repo:
    docs/databricks-performance-monitoring.md

Usage:

    from timing import timed, log_summary

    spans = {}
    with timed("history_read", spans):
        ...            # the work to measure
    with timed("generate", spans):
        ...
    log_summary(spans)   # -> [TIMING] total=1234ms | history_read=40ms(3%) generate=1150ms(93%) ...

To actually SEE these lines in the Databricks serving Logs, two things must be true
(both fail silently otherwise - see the performance-monitoring doc's "Two gotchas"):
  1. Your log records must reach the ROOT logger (the container captures root). Do NOT
     set propagate=False on a custom logger with its own handler.
  2. The level must be enabled - it is not by default. Lower root's level, wired to an
     endpoint env var, e.g. LOG_LEVEL=DEBUG (off / INFO in production).
"""

from __future__ import annotations

import contextlib
import functools
import logging
import time

log = logging.getLogger("timing")


def timed_call(label=None, level=logging.DEBUG, logger=None):
    """Decorator: log the wrapped function's duration.

    Put it in front of each expensive call in the real service - the SQL reads, the
    connection setup, each LLM call - to see where a request's time goes:

        @timed_call()
        def fetch_history(...): ...

        @timed_call("input_guardrail")
        def classify(...): ...

    Logs at DEBUG by default (gate it with a LOG_LEVEL env var, off in production).
    Standard library only; works on Python 3.9 through 3.12.
    """

    def decorator(fn):
        name = label or fn.__name__
        lg = logger or log

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                lg.log(level, "[TIMING] %s took %.1f ms", name, (time.perf_counter() - start) * 1000.0)

        return wrapper

    return decorator


@contextlib.contextmanager
def timed(label, spans):
    """Time the wrapped block and record its milliseconds under `label` in `spans` (a dict)."""

    start = time.perf_counter()
    try:
        yield
    finally:
        spans[label] = (time.perf_counter() - start) * 1000.0


def log_summary(spans, logger=None):
    """Emit one line: total plus each phase's ms and its percentage of the total."""

    logger = logger or log
    total = sum(spans.values()) or 1.0
    parts = " ".join(f"{label}={ms:.0f}ms({100 * ms / total:.0f}%)" for label, ms in spans.items())
    logger.info("[TIMING] total=%.0fms | %s", total, parts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo = {}
    with timed("step_a", demo):
        time.sleep(0.05)
    with timed("step_b", demo):
        time.sleep(0.15)
    log_summary(demo)
