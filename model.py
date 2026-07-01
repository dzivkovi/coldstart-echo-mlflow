"""Minimal scale-to-zero cold-start harness (MLflow pyfunc), self-contained.

Echoes the request text and appends a random offline fortune. No artifacts, no
network, no file reads at load time - fortunes are embedded inline so the serving
container has nothing external to resolve. The only latency it adds over a warm
call is the platform cold start (waking a replica), which is what we measure.

Logged via MLflow "models from code" (set_model at the bottom) so the serving
container loads THIS file as the model definition.
"""

from __future__ import annotations

import random
from typing import Any

import mlflow.models
import mlflow.pyfunc
import pandas as pd

# Provenance is deliberately clean for corporate / OSS review: original lines plus
# short PUBLIC-DOMAIN (pre-1929) quotations. NOT the Unix fortunes database.
FORTUNES = [
    "Simplicity is the ultimate sophistication.",
    "A slow system that tells the truth beats a fast one that lies.",
    "Make it work, make it right, make it fast - in that order.",
    "The cheapest, fastest, most reliable component is the one that is not there.",
    "Cold starts are honest: the wait is the replica waking, not the model thinking.",
    "Name the real problem and half of it is solved.",
    "When in doubt, instrument before you assert.",
    "A good API tells you what happened, not just that something did.",
    "Patience is warm; scale-to-zero is cheap. Choose per environment.",
    "Under-promise in the standup; over-deliver with the numbers.",
    "Concede the precise fact and you keep the larger point.",
    "You have power over your mind, not outside events. - Marcus Aurelius",
    "The impediment to action advances action. What stands in the way becomes the way. - Marcus Aurelius",
    "A journey of a thousand miles begins with a single step. - Lao Tzu",
    "Nature does not hurry, yet everything is accomplished. - Lao Tzu",
    "We suffer more often in imagination than in reality. - Seneca",
    "Luck is what happens when preparation meets opportunity. - Seneca",
    "It does not matter how slowly you go as long as you do not stop. - Confucius",
    "It is not what happens to you, but how you react to it that matters. - Epictetus",
    "We are what we repeatedly do. Excellence, then, is a habit. - Aristotle",
    "No man ever steps in the same river twice. - Heraclitus",
    "The wound is the place where the light enters you. - Rumi",
    "He who has a why to live can bear almost any how. - Friedrich Nietzsche",
]


class EchoFortuneModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
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
            out.append({"echo": echo, "fortune": random.choice(FORTUNES)})
        return out


mlflow.models.set_model(EchoFortuneModel())
