"""Log + register the echo/fortune pyfunc to Unity Catalog, ready for serving.

Auth uses the Databricks CLI profile (set DATABRICKS_CONFIG_PROFILE, default
'coldstart'). Registration target is UC: <catalog>.<schema>.<name>.

Run:  DATABRICKS_CONFIG_PROFILE=coldstart python register_model.py
Then create the scale-to-zero serving endpoint (see README / endpoint_config.json).
"""

from __future__ import annotations

import os

import mlflow
import pandas as pd

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# On Databricks tracking you must target a workspace experiment (a /Users/<you>/...
# path) before starting a run, or start_run fails with experiment ID None.
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
if EXPERIMENT:
    mlflow.set_experiment(EXPERIMENT)

CATALOG = os.environ.get("UC_CATALOG", "workspace")
SCHEMA = os.environ.get("UC_SCHEMA", "default")
NAME = os.environ.get("MODEL_NAME", "coldstart_echo_fortune")
REGISTERED = f"{CATALOG}.{SCHEMA}.{NAME}"

input_example = pd.DataFrame([{"prompt": "hello from the cold-start harness"}])

with mlflow.start_run(run_name="coldstart-echo-fortune"):
    info = mlflow.pyfunc.log_model(
        name="model",  # MLflow 3.x; on 2.x use artifact_path="model"
        python_model="model.py",  # models-from-code: a PATH, not an instance
        artifacts={"fortunes": "fortunes.txt"},
        input_example=input_example,
        pip_requirements=["mlflow", "pandas"],
        registered_model_name=REGISTERED,
    )

print("Registered:", REGISTERED)
print("Model URI:", info.model_uri)
print("Next: create a scale-to-zero serving endpoint from this model (see README).")
