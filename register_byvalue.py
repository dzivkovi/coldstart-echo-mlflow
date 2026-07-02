"""The echo/fortune model, and a script to register it to Unity Catalog.

Importing this module gives you the model class with NO side effects - safe for a
local smoke test (see the README). Running it as a script registers the model to
Databricks.

The model is logged BY VALUE (the class instance is pickled), so there is no source
file path baked into the model. That is the detail that lets it load on the Linux
serving container no matter which OS you register from - the reason this approach
works where logging a code file did not.

Register (from any machine with the Databricks CLI authenticated):

    DATABRICKS_HOST=https://<host> DATABRICKS_TOKEN=<token> \\
    MLFLOW_EXPERIMENT=/Users/<you>/coldstart-echo-fortune \\
    python register_byvalue.py

See DEPLOYMENT.md for how to get the token and why these env vars are needed.
"""

from __future__ import annotations

import logging
import os
import random
import time

import mlflow.pyfunc
import pandas as pd


class EchoFortuneModel(mlflow.pyfunc.PythonModel):
    # LOG-ONLY threshold: a request slower than this gets ONE warning line after it
    # finishes. It NEVER terminates, times out, or interrupts the request - the caller
    # always gets the full response; this only decides whether the call was worth logging.
    # 8s reflects the old web "8-second rule" (max-tolerable time-to-first-byte). The echo's own
    # work is sub-millisecond, so without the demo slow-sim below this would never trip; it is here
    # as the copy-into-the-real-service pattern (wrap the SQL reads / LLM calls and this surfaces
    # the slow ones).
    SLOW_REQUEST_LOG_THRESHOLD_SECONDS = 8.0

    # DEMO pipeline simulation: stand-ins for the utilities a real predict() calls before it can
    # answer - a connection/pool setup, a couple of point reads, a retrieval step. Each has a tiny
    # baseline cost AND can independently SPIKE (a slow warehouse read, a cold connection pool, a
    # slow downstream call), which is what trips "[REQUEST] slow". Each contributes its OWN timing
    # span, so the breakdown shows WHICH utility was slow - the whole point of instrumentation.
    # Generic names on purpose; a real service measures its own steps and never injects latency -
    # delete this simulation when you copy the pattern. Set STEP_SPIKE_PROBABILITY = 0.0 to disable.
    SIMULATED_STEPS = ("connection_setup", "reference_lookup", "history_fetch", "retrieval")
    STEP_BASELINE_SECONDS = (0.003, 0.02)   # per-step baseline latency (3-20ms): realistic and tiny
    STEP_SPIKE_PROBABILITY = 0.10           # each step spikes independently; ~1 in 3 requests slow
    STEP_SPIKE_SECONDS = 9.0                # a spike sleeps past SLOW_REQUEST_LOG_THRESHOLD_SECONDS

    # Provenance is deliberately clean for corporate / open-source review: every
    # line is either original to this repo or a short PUBLIC-DOMAIN (pre-1929)
    # quotation. This is intentionally NOT the Unix `fortunes` database, whose
    # individual entries have undocumented provenance.
    FORTUNES = [
        "Simplicity is the ultimate sophistication.",
        "Cold starts are honest: the wait is the replica waking, not the model thinking.",
        "Measure twice, cut once.",
        "A slow system that tells the truth beats a fast one that lies.",
        "You have power over your mind, not outside events. - Marcus Aurelius",
        "A journey of a thousand miles begins with a single step. - Lao Tzu",
        "We suffer more often in imagination than in reality. - Seneca",
        "It does not matter how slowly you go as long as you do not stop. - Confucius",
        "We are what we repeatedly do. Excellence, then, is a habit. - Aristotle",
        "The wound is the place where the light enters you. - Rumi",
    ]

    def load_context(self, context):
        # The serving container captures the ROOT logger (note the "WARNING:root:" lines
        # in the endpoint Logs), so route timing through root and lower its level enough
        # for DEBUG to emit. Gate with LOG_LEVEL.
        #
        # TODO (before reusing this in a real service): this DEMO defaults to DEBUG on purpose, so
        # the per-request [TIMING] lines appear with zero setup - otherwise people set nothing, see
        # no logs, and go hunting in Splunk. For a real deployment set LOG_LEVEL explicitly (INFO is
        # the usual choice): at INFO the per-request timing goes quiet and only the WARNING-level
        # cold-start / slow / error lines remain.
        import sys

        level = getattr(logging, os.environ.get("LOG_LEVEL", "DEBUG").upper(), logging.DEBUG)
        root = logging.getLogger()
        root.setLevel(level)
        if not root.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
            root.addHandler(handler)
        for handler in root.handlers:
            handler.setLevel(level)
        logging.getLogger("echo").setLevel(level)  # propagates to root (default)

        # COLD-START MARKER. load_context runs ONCE per worker process boot. We do NOT log a
        # worker_boot line here - a boot with no traffic is not interesting, and from inside the
        # model "seconds since boot" is a poor cold-start proxy anyway. Instead predict() logs the
        # FIRST request this worker serves, tagged with that request's LATENCY (the thing that
        # actually hurt). Plain boolean (no lock): a Lock is not picklable and MLflow logs the
        # model by value, so a lock would break serialization.
        self._saw_first_request = False

    def predict(self, context, model_input, params=None):
        # INSTRUMENTATION PATTERN - copy this style into the real service, in front of the
        # SQL reads, the connection setup, and each LLM call. The echo does almost nothing,
        # so these numbers are tiny; the point is the STYLE, not the values. In an importable
        # module you would use the @timed_call decorator from timing.py instead of inline
        # blocks (the echo is pickled by value, so it stays self-contained here).
        #
        # TWO-TIER LOGGING (this is the shape to copy into the real service):
        #   - DEBUG: EVERY request logs one [TIMING] line - the per-step breakdown. This is the
        #     instrumentation recipe to reuse: measure each step, report its share of the total.
        #   - WARNING (still shows when DEBUG is off, e.g. production at INFO): the cold-first
        #     request once per worker, any error, and any call slower than the threshold.
        # The cold-first line carries the breakdown itself, so that one request logs a SINGLE line -
        # never a [COLDSTART] + [TIMING] pair. Every later warm request logs just the [TIMING] line.
        log = logging.getLogger("echo")

        started = time.perf_counter()
        # Consume the cold-first marker BEFORE doing the work, so a failure on the first request still
        # counts as cold-first (the next request is warm, not a second "cold" line). BEST-EFFORT per
        # worker, not exact-once: no lock (a Lock is not picklable under by-value logging), so if two
        # requests race this very first check you may get two [COLDSTART] lines - fine for a marker.
        # getattr guards the (unexpected) case of predict() running before load_context().
        is_first = not getattr(self, "_saw_first_request", False)
        if is_first:
            self._saw_first_request = True

        spans = {}
        error = None
        try:
            t = time.perf_counter()
            if isinstance(model_input, pd.DataFrame):
                rows = model_input.to_dict(orient="records")
            elif isinstance(model_input, dict):
                rows = [model_input]
            else:
                rows = model_input
            spans["parse_input"] = (time.perf_counter() - t) * 1000.0

            # DEMO pipeline: stand-ins for the utilities a real predict() calls before it answers.
            # Each has a tiny baseline cost and can independently SPIKE (slow warehouse read, cold
            # connection pool, slow downstream call) - that is what trips "[REQUEST] slow". Each is
            # its OWN span, so the breakdown pinpoints WHICH utility was slow. Delete when you copy.
            for step in self.SIMULATED_STEPS:
                t = time.perf_counter()
                time.sleep(random.uniform(*self.STEP_BASELINE_SECONDS))
                if random.random() < self.STEP_SPIKE_PROBABILITY:
                    time.sleep(self.STEP_SPIKE_SECONDS)
                spans[step] = (time.perf_counter() - t) * 1000.0

            t = time.perf_counter()
            out = []
            for r in rows:
                echo = (r.get("prompt") or r.get("query") or str(r)) if isinstance(r, dict) else str(r)
                out.append({"echo": echo, "fortune": random.choice(self.FORTUNES)})
            spans["build_answer"] = (time.perf_counter() - t) * 1000.0

            return out
        except Exception as exc:
            error = exc
            raise
        finally:
            latency_ms = (time.perf_counter() - started) * 1000.0
            threshold_ms = self.SLOW_REQUEST_LOG_THRESHOLD_SECONDS * 1000.0
            # Do the logging work only if a line will actually come out: the cold-first request, an
            # error, a slow call (all WARNING), OR any request while DEBUG is on (the per-request
            # [TIMING] line). A warm success with DEBUG off - production at INFO - skips it entirely.
            if is_first or error is not None or latency_ms >= threshold_ms or log.isEnabledFor(logging.DEBUG):
                total = sum(spans.values()) or 1.0
                breakdown = " ".join(
                    f"{k}={spans[k]:.3f}ms({100 * spans[k] / total:.0f}%)" for k in spans
                )
                if is_first:
                    # Cold-first: WARNING (shows at any level), with the breakdown baked in, so this
                    # one request logs a single line - no separate [TIMING]. latency_ms is the cold
                    # signal; the true caller-felt wait is measured OUTSIDE at the facade.
                    log.warning(
                        "[COLDSTART] first request after boot: pid=%d latency_ms=%.1f status=%s | %s",
                        os.getpid(), latency_ms, "error" if error else "ok", breakdown,
                    )
                elif error is not None:
                    log.warning(
                        "[REQUEST] error latency_ms=%.1f error_type=%s | %s",
                        latency_ms, type(error).__name__, breakdown,
                    )
                elif latency_ms >= threshold_ms:
                    # Slow: WARNING, so it surfaces even in production where the DEBUG timing is off.
                    log.warning(
                        "[REQUEST] slow latency_ms=%.1f threshold_ms=%.1f | %s",
                        latency_ms, threshold_ms, breakdown,
                    )
                else:
                    # Every normal warm request (DEBUG only): the per-step timing breakdown - the
                    # copy-paste recipe your own SQL reads / pool waits / LLM calls would use.
                    log.debug(
                        "[TIMING] total=%.3fms latency_ms=%.1f | %s", total, latency_ms, breakdown,
                    )


def _register() -> None:
    """Log + register the model to Unity Catalog. Talks to Databricks; run as a script."""

    import mlflow

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    experiment = os.environ.get("MLFLOW_EXPERIMENT")
    if experiment:
        mlflow.set_experiment(experiment)

    catalog = os.environ.get("UC_CATALOG", "workspace")
    schema = os.environ.get("UC_SCHEMA", "default")
    model = f"{catalog}.{schema}.coldstart_echo_fortune"

    with mlflow.start_run(run_name="coldstart-echo-fortune"):
        info = mlflow.pyfunc.log_model(
            artifact_path="model",  # MLflow 2.x; on 3.x this argument is named `name`
            python_model=EchoFortuneModel(),
            input_example=pd.DataFrame([{"prompt": "hello"}]),
            pip_requirements=["mlflow", "pandas"],
            registered_model_name=model,
        )

    print("Registered", model, "version", info.registered_model_version)
    print("Next: set entity_version in endpoint_config.json to this version, then create the endpoint.")


if __name__ == "__main__":
    _register()
