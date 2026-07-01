# coldstart-echo-mlflow

A deliberately tiny MLflow model you deploy as a **scale-to-zero Databricks Model Serving endpoint**, so you can reproduce and **measure a real cold start**. It just echoes your request and adds a random offline quote - no model weights, no network calls. The only time it adds over a warm call is the platform cold start (a replica waking up), which is exactly what we want to measure.

New to MLflow or Databricks? Start with **Run it locally** below - that part needs no Databricks account at all.

## Why this exists (and why it is NOT a Databricks App)

- A **Databricks App** stays warm while it is running - it has no per-request cold start, so it is useless for measuring one.
- A **Model Serving endpoint** with `scale_to_zero_enabled=true` suspends after it sits idle and cold-starts on the first request. That is the behaviour a real request-router serving endpoint shows, because such a router is also a scale-to-zero pyfunc model.
- The cold start is **platform behaviour**: it is the replica waking, not your code running. A six-line echo cold-starts the same as a heavy service. That is the whole reason this trivial harness is valid - keep it tiny, and you are measuring the platform, not the model.

Companion projects: `genai-coldstart-guard` (the cold-start classification facade + the endpoint-state model) and `coldstart-guard-databricks-app` (a web-app version, which does NOT cold-start).

## Files

- `register_byvalue.py` - the model class, and the script that registers it to Unity Catalog.
- `endpoint_config.json` - the serving-endpoint config (scale-to-zero, smallest size). See the hazard note under Deploy.
- `databricks_notebook.py` - an alternative walkthrough for a standard (paid) workspace. It does **not** work on Free Edition - read its header first.
- `requirements.txt` - `mlflow` (2.x), `pandas`.

## Run it locally (no Databricks, any operating system)

Just to confirm the model works. No account, no cloud, no Linux required:

```bash
pip install -r requirements.txt
python -c "import pandas as pd; from register_byvalue import EchoFortuneModel; print(EchoFortuneModel().predict(None, pd.DataFrame([{'prompt':'hello'}])))"
```

You should see something like `[{'echo': 'hello', 'fortune': '...'}]`. That proves the model is fine. You cannot reproduce a cold start locally, though - your own machine never scales to zero. For that you need the endpoint below.

## Deploy to Databricks (the scale-to-zero endpoint)

This is a **minimal happy path in three phases, then a wait**, not a one-liner. The full runbook - how to get the auth token, and the gotchas that bit us - is in [DEPLOYMENT.md](DEPLOYMENT.md). The short shape:

1. **Register** the model to Unity Catalog: run `register_byvalue.py` (from any machine with the Databricks CLI authenticated - any OS).
2. **Create** the endpoint: `databricks serving-endpoints create --json @endpoint_config.json`.
3. **Wait ~10 minutes**, poll until it reports `READY`, then query it.

> HAZARD: `endpoint_config.json` pins `entity_version` (currently `11`). After you register, change it to the version number your registration step printed - otherwise you deploy an old or wrong model version.

## Measure a cold start

```bash
# Warm call - time it:
time databricks serving-endpoints query coldstart-echo-fortune \
  --json '{"dataframe_records":[{"prompt":"hello"}]}' --profile <profile>

# Let it sit idle a while (minutes) so it scales to zero, then call again and time it.
# cold latency - warm latency = your cold-start cost.
```

Note whether the first cold hit comes back as a fast `429` ("starting"), a slow `200`, or a timeout - that surface is what the `genai-coldstart-guard` facade classifies as "warming / please hold."

## Cost

Scale-to-zero means it costs nothing while idle - that is the point. When you are finished, delete it:

```bash
databricks serving-endpoints delete coldstart-echo-fortune --profile <profile>
```
