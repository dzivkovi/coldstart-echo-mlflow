"""Register the echo model by VALUE (no code path, no module import).

The class is defined here in __main__ and logged as an INSTANCE, so cloudpickle
serializes the class BY VALUE into python_model.pkl. There is no model_code_path
(the Windows-absolute-path problem) and no `import model` at serving time (the
by-reference problem). Fortunes are a class attribute so they are captured too.

Run from the laptop (the only identity that can write the model to UC on Free
Edition):
  DATABRICKS_HOST=... DATABRICKS_TOKEN=... MLFLOW_EXPERIMENT=/Users/<you>/... python register_byvalue.py
"""

from __future__ import annotations

import os

import mlflow
import mlflow.pyfunc
import pandas as pd

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
EXP = os.environ.get("MLFLOW_EXPERIMENT")
if EXP:
    mlflow.set_experiment(EXP)

CATALOG = os.environ.get("UC_CATALOG", "workspace")
SCHEMA = os.environ.get("UC_SCHEMA", "default")
MODEL = f"{CATALOG}.{SCHEMA}.coldstart_echo_fortune"


class EchoFortuneModel(mlflow.pyfunc.PythonModel):
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
        import random

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


if __name__ == "__main__":
    with mlflow.start_run(run_name="coldstart-echo-byvalue"):
        info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=EchoFortuneModel(),
            input_example=pd.DataFrame([{"prompt": "hello"}]),
            pip_requirements=["mlflow", "pandas"],
            registered_model_name=MODEL,
        )
    print("VERSION", info.registered_model_version)
