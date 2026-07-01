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
import threading
import time

import mlflow.pyfunc
import pandas as pd


class EchoFortuneModel(mlflow.pyfunc.PythonModel):
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
        # for DEBUG to emit. Gate with LOG_LEVEL; default DEBUG here so the demo shows.
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

        # COLD-START MARKER. load_context runs ONCE per worker process boot, so this line marks
        # a worker that just cold-started - per WORKER, not per replica (Databricks runs several
        # gunicorn workers, each loads the model). Databricks gives no built-in cold-start signal,
        # so we print our own. WARNING so it is always visible; monotonic clock for elapsed time.
        self._boot_monotonic = time.monotonic()
        self._request_count = 0
        self._request_lock = threading.Lock()
        logging.getLogger("echo").warning("[COLDSTART] worker_boot pid=%d", os.getpid())

    def predict(self, context, model_input, params=None):
        # INSTRUMENTATION PATTERN - copy this style into the real service, in front of the
        # SQL reads, the connection setup, and each LLM call. The echo does almost nothing,
        # so these numbers are tiny; the point is the STYLE, not the values. In an importable
        # module you would use the @timed_call decorator from timing.py instead of inline
        # blocks (the echo is pickled by value, so it stays self-contained here).
        log = logging.getLogger("echo")

        # Flag the first request THIS worker serves after boot - that is the cold-started one.
        with self._request_lock:
            self._request_count += 1
            request_number = self._request_count
        cold_first = request_number == 1
        log.debug(
            "[COLDSTART] request pid=%d request_number=%d cold_first_request=%s seconds_since_boot=%.3f",
            os.getpid(), request_number, cold_first, time.monotonic() - self._boot_monotonic,
        )

        spans = {}

        t = time.perf_counter()
        if isinstance(model_input, pd.DataFrame):
            rows = model_input.to_dict(orient="records")
        elif isinstance(model_input, dict):
            rows = [model_input]
        else:
            rows = model_input
        spans["parse_input"] = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        out = []
        for r in rows:
            echo = (r.get("prompt") or r.get("query") or str(r)) if isinstance(r, dict) else str(r)
            out.append({"echo": echo, "fortune": random.choice(self.FORTUNES)})
        spans["build_answer"] = (time.perf_counter() - t) * 1000.0

        total = sum(spans.values()) or 1.0
        log.debug(
            "[TIMING] total=%.3fms | " + " ".join(f"{k}=%.3fms(%.0f%%)" for k in spans),
            total, *[x for k in spans for x in (spans[k], 100 * spans[k] / total)],
        )
        return out


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
