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

import os
import random

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

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            rows = model_input.to_dict(orient="records")
        elif isinstance(model_input, dict):
            rows = [model_input]
        else:
            rows = model_input
        out = []
        for r in rows:
            echo = (r.get("prompt") or r.get("query") or str(r)) if isinstance(r, dict) else str(r)
            out.append({"echo": echo, "fortune": random.choice(self.FORTUNES)})
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
