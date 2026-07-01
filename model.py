"""Minimal scale-to-zero cold-start harness (MLflow pyfunc).

Echoes the request text and appends a random offline fortune. No model weights,
no network, no dependencies beyond mlflow/pandas - so the ONLY latency it adds
over a warm call is the platform cold start (waking a replica from zero).

Deploy it as a Databricks Model Serving endpoint with scale_to_zero_enabled=true
to reproduce and MEASURE the cold start that any scale-to-zero pyfunc shows -
including the office dispatcher. The delay is the replica waking, not what's
inside, which is the whole point: a six-line echo cold-starts the same as a
heavy app.
"""

from __future__ import annotations

import random
from pathlib import Path

import mlflow.pyfunc
import pandas as pd


class EchoFortuneModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        text = Path(context.artifacts["fortunes"]).read_text(encoding="utf-8")
        self._fortunes = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.startswith("#")
        ]

    def _fortune(self) -> str:
        return random.choice(self._fortunes) if self._fortunes else ""

    def predict(self, context, model_input, params=None):
        # Model Serving hands us a DataFrame (from dataframe_records/split);
        # be tolerant of dict/list too so it is easy to call by hand.
        if isinstance(model_input, pd.DataFrame):
            rows = model_input.to_dict(orient="records")
        elif isinstance(model_input, dict):
            rows = [model_input]
        elif isinstance(model_input, list):
            rows = model_input
        else:
            rows = [{"input": str(model_input)}]

        out = []
        for row in rows:
            if isinstance(row, dict):
                echo = row.get("prompt") or row.get("text") or row.get("query") or str(row)
            else:
                echo = str(row)
            out.append({"echo": echo, "fortune": self._fortune()})
        return out


# MLflow "models from code": the serving container loads THIS file as the model
# definition, so the class is always importable. Logging a class INSTANCE instead
# pickles it by reference and fails at serving with "failed to load the model".
import mlflow.models  # noqa: E402

mlflow.models.set_model(EchoFortuneModel())
