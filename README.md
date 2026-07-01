# coldstart-echo-mlflow

A deliberately tiny MLflow pyfunc you deploy as a **scale-to-zero Model Serving endpoint** to reproduce and **measure** Databricks cold starts. It echoes your request and appends a random offline fortune - no weights, no network. The only latency it adds over a warm call is the platform cold start (waking a replica), which is exactly what we want to measure.

## Why this exists (and why it is NOT a Databricks App)

- A **Databricks App** stays warm while Running - it has no per-request cold start, so it is useless for measuring one.
- A **Model Serving endpoint** with `scale_to_zero_enabled=true` suspends after idle and cold-starts on the first request. That is the behavior the office dispatcher shows, because the dispatcher is also a scale-to-zero pyfunc.
- The cold start is **platform behavior**: it is the replica waking, not the code running. So a six-line echo cold-starts the same as a heavy app. That is the whole reason this harness is valid: keep it trivial, measure the platform.

Related: the `genai-coldstart-guard` POC (the classification facade + the endpoint-state model) and the `coldstart-guard-databricks-app` bundle (the web-app version, which does NOT cold-start).

## Files

- `model.py` - the `EchoFortuneModel` pyfunc.
- `fortunes.txt` - offline quotes (the model's only artifact).
- `register_model.py` - logs + registers the model to Unity Catalog.
- `endpoint_config.json` - serving endpoint config (scale-to-zero, Small).
- `requirements.txt` - `mlflow`, `pandas`.

## Deploy (Databricks CLI + MLflow)

Prereqs: Databricks CLI authenticated (`databricks auth login --host <host> --profile <profile>`), and `pip install -r requirements.txt`.

```bash
# 1. Register the model to Unity Catalog (edit UC_CATALOG/UC_SCHEMA if needed)
DATABRICKS_CONFIG_PROFILE=<profile> UC_CATALOG=workspace UC_SCHEMA=default python register_model.py

# 2. Create the scale-to-zero serving endpoint from the registered model
databricks serving-endpoints create --json @endpoint_config.json --profile <profile>

# 3. Wait until it is READY
databricks serving-endpoints get coldstart-echo-fortune --profile <profile>
```

If `entity_version` in `endpoint_config.json` does not match (first registration is version 1), update it to the version `register_model.py` printed.

## Measure a cold start

```bash
# Warm call - time it
time databricks serving-endpoints query coldstart-echo-fortune \
  --json '{"dataframe_records":[{"prompt":"hello"}]}' --profile <profile>

# Let it scale to zero (idle a while - minutes), then call again and time it.
# The difference between the cold call and the warm call is your cold-start cost.
```

Record: warm latency, cold latency, and whether the first cold hit returns a fast 429 ("starting"), a slow 200, or a timeout. That surface is what the facade classifies as "warming / please hold."

## Cost control

Scale-to-zero means it costs nothing while idle (that is the point). When you are done, delete it:

```bash
databricks serving-endpoints delete coldstart-echo-fortune --profile <profile>
```

## Notes / gotchas

- `mlflow.pyfunc.log_model` uses `artifact_path=` on MLflow 2.x; on MLflow 3.x that argument is renamed `name=`. Adjust `register_model.py` if you are on 3.x.
- Registration needs a Unity Catalog schema you can write to; `workspace.default` is the Free Edition default. In the office, use a catalog/schema you own.
- Whether **custom** model serving is available depends on the workspace tier. Foundation Model endpoints are always present; custom pyfunc serving may be gated on some tiers.
