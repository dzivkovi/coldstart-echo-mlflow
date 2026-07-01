# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy the scale-to-zero echo/fortune endpoint (STANDARD WORKSPACE ONLY)
# MAGIC
# MAGIC ## Read this first
# MAGIC
# MAGIC **This notebook does NOT work on Databricks Free Edition.** On Free Edition the
# MAGIC serverless compute that runs a notebook is explicitly denied permission to save a
# MAGIC model to storage, so the first cell fails with an S3 `AccessDenied` error. On Free
# MAGIC Edition, register from a local machine instead - see `register_byvalue.py` and
# MAGIC `DEPLOYMENT.md`.
# MAGIC
# MAGIC On a **standard (paid) workspace**, this notebook is the simplest path: it runs as
# MAGIC you, on Databricks compute, already authenticated - so there is no token juggling.
# MAGIC Two cells: (1) register the model, (2) create the scale-to-zero endpoint. Then watch
# MAGIC it build under **Serving** in the left nav.

# COMMAND ----------

# ---- Cell 1: register the model to Unity Catalog ----
import random

import mlflow
import pandas as pd

# Change these for your workspace (use a catalog/schema you own):
CATALOG = "workspace"
SCHEMA = "default"
MODEL = f"{CATALOG}.{SCHEMA}.coldstart_echo_fortune"

# Provenance is deliberately clean for corporate / open-source review: original lines
# plus short PUBLIC-DOMAIN (pre-1929) quotations only. NOT the Unix fortunes database.
FORTUNES = [
    "Simplicity is the ultimate sophistication.",
    "Cold starts are honest: the wait is the replica waking, not the model thinking.",
    "Measure twice, cut once.",
    "A journey of a thousand miles begins with a single step. - Lao Tzu",
    "We suffer more often in imagination than in reality. - Seneca",
    "It does not matter how slowly you go as long as you do not stop. - Confucius",
    "We are what we repeatedly do. Excellence, then, is a habit. - Aristotle",
    "The wound is the place where the light enters you. - Rumi",
]


class EchoFortuneModel(mlflow.pyfunc.PythonModel):
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
            out.append({"echo": echo, "fortune": random.choice(FORTUNES)})
        return out


mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name="coldstart-echo-fortune"):
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=EchoFortuneModel(),
        input_example=pd.DataFrame([{"prompt": "hello"}]),
        pip_requirements=["mlflow", "pandas"],
        registered_model_name=MODEL,
    )

VERSION = info.registered_model_version
print(f"Registered {MODEL} version {VERSION}")

# COMMAND ----------

# ---- Cell 2: create (or update) the scale-to-zero serving endpoint ----
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

ENDPOINT = "coldstart-echo-fortune"

w = WorkspaceClient()
served = [
    ServedEntityInput(
        entity_name=MODEL,
        entity_version=str(VERSION),
        workload_size="Small",
        scale_to_zero_enabled=True,  # the whole point: it goes cold when idle
    )
]

existing = [e.name for e in (w.serving_endpoints.list() or [])]
if ENDPOINT in existing:
    print(f"Endpoint '{ENDPOINT}' exists - updating to version {VERSION}...")
    w.serving_endpoints.update_config(name=ENDPOINT, served_entities=served)
else:
    print(f"Creating endpoint '{ENDPOINT}' on {MODEL} v{VERSION}...")
    w.serving_endpoints.create(name=ENDPOINT, config=EndpointCoreConfigInput(served_entities=served))

print("Now watch it build under Serving in the left nav (~10 min).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test it once it is READY
# MAGIC
# MAGIC Use the **Query endpoint** button on the Serving page, or run the cell below.

# COMMAND ----------

# ---- Cell 3: run this again AFTER the endpoint shows READY under Serving ----
try:
    resp = w.serving_endpoints.query(
        name="coldstart-echo-fortune",
        dataframe_records=[{"prompt": "hello from the notebook"}],
    )
    print(resp.predictions)
except Exception as e:
    print("Not ready yet - this is expected right after creation.")
    print("Re-run THIS cell once the endpoint shows READY under Serving (~10 min).")
    print("detail:", str(e)[:200])
