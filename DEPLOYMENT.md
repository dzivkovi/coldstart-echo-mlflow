# Deploying the scale-to-zero cold-start harness (MLflow Model Serving)

A battle-tested runbook - every command here was actually run and worked (AWS Free Edition). Generic by design: no secrets, no proprietary names. This deploys a minimal MLflow pyfunc as a scale-to-zero Model Serving endpoint so you can measure a real cold start. It is NOT a Databricks App (Apps stay warm and cannot cold-start).

## Prerequisites

- Databricks CLI authenticated: `databricks auth login --host https://<host> --profile <profile>`.
- Python with MLflow: `pip install -r requirements.txt` (mlflow, pandas). NOTE: MLflow 3.x is assumed.
- A Unity Catalog schema you can write to (Free Edition default: `workspace.default`).

## The working sequence

```bash
# Pin the CLI and pull a short-lived OAuth token (see GOTCHA 1 for why).
DB=/c/Users/<you>/AppData/Local/Microsoft/WinGet/Links/databricks   # or `command -v databricks`
PROFILE=<profile>
TOKEN=$("$DB" auth token -p "$PROFILE" | grep -oE '"access_token": *"[^"]+"' | sed 's/.*: *"//;s/"//')

# 1. Register the pyfunc to Unity Catalog. MLflow auth is via host+token, NOT the profile (GOTCHA 1).
export DATABRICKS_HOST=https://<host>
export DATABRICKS_TOKEN=$TOKEN
unset DATABRICKS_CONFIG_PROFILE
export UC_CATALOG=workspace UC_SCHEMA=default
export MLFLOW_EXPERIMENT=/Users/<your-email>/coldstart-echo-fortune   # GOTCHA 2
python register_model.py        # prints the registered name + version

# 2. Create the scale-to-zero serving endpoint (uses the CLI profile, not the env token).
MSYS_NO_PATHCONV=1 "$DB" serving-endpoints create --json @endpoint_config.json -p "$PROFILE"
#   ^ this call BLOCKS waiting for readiness and may "time out" in a shell after ~2 min.
#     That is fine - the endpoint is still building. Do NOT re-run it. Poll with `get` instead:

# 3. Poll until ready (custom model serving builds a container + loads the model: several minutes).
until "$DB" serving-endpoints get coldstart-echo-fortune -p "$PROFILE" | grep -q '"ready": "READY"'; do
  echo "building..."; sleep 20
done
```

## Gotchas (all three actually bit us)

1. MLflow + Databricks U2M OAuth. MLflow could NOT use the OAuth CLI profile directly - it failed to build the SDK workspace client and fell back to no credentials, returning `401: Credential was not sent or was of an unsupported type`. The clean fix WITHOUT creating a PAT: pull the short-lived OAuth access token with `databricks auth token -p <profile>` and hand it to MLflow via `DATABRICKS_HOST` + `DATABRICKS_TOKEN` (and `unset DATABRICKS_CONFIG_PROFILE` so it does not fight the env token). The token is a ~1-hour bearer, plenty for a registration.
2. Experiment path required. On Databricks tracking, `mlflow.start_run()` fails with `RESOURCE_DOES_NOT_EXIST: Could not find experiment with ID None` unless you first set a workspace experiment. `register_model.py` calls `mlflow.set_experiment(MLFLOW_EXPERIMENT)` where the path is `/Users/<your-email>/<name>`.
3. MLflow 3.x renamed the arg. `mlflow.pyfunc.log_model(artifact_path=...)` became `name=...` in MLflow 3.x. `register_model.py` uses `name="model"`. On MLflow 2.x, change it back to `artifact_path="model"`.
4. Log the model FROM CODE, not as an instance. Passing a class instance (`python_model=EchoFortuneModel()`) pickles the class by reference; the serving container then cannot `import model` and fails at load with "Model server failed to load the model" (endpoint reaches UPDATE_FAILED after the build). Fix (MLflow's blessed pattern, and how the office model is packaged): end `model.py` with `mlflow.models.set_model(EchoFortuneModel())`, and log with `python_model="model.py"` (the PATH). This is exactly what broke v1 and fixed v2 here.

## WHAT ACTUALLY WORKED (use register_byvalue.py)

Resolved 2026-07-01 on AWS Free Edition after several failed attempts. The winning recipe:

- Register from the LAPTOP (not serverless), as a BY-VALUE instance: use `register_byvalue.py`, which defines the model class in `__main__` and logs the INSTANCE. cloudpickle serializes the class by value into `python_model.pkl`, so there is NO `model_code_path` at all - which is what broke every other attempt. Then create the endpoint from `endpoint_config.json`. It built to READY and serves; Python 3.12 turned out to be fine.
- Verified: `serving-endpoints query ... {"dataframe_records":[{"prompt":"hi"}]}` returns `{"predictions":[{"echo":"hi","fortune":"..."}]}`, and the facade app (BACKEND_MODE=databricks) returns the answer end-to-end.

### Two dead ends we ruled out (so you do not repeat them)

1. models-from-code / class-instance-by-reference from Windows: the `MLmodel` recorded `model_code_path: C:\Users\...\model.py` (an absolute Windows path the Linux serving container cannot resolve) -> `UPDATE_FAILED`, "Model server failed to load the model." By-value logging (register_byvalue.py) removes the code path entirely and fixes this. It was the Windows PATH, not the Python version.
2. Running the notebook on Free Edition serverless (job OR interactive): the serverless compute role is EXPLICITLY DENIED writing model artifacts to UC storage (`s3:PutObject ... AccessDenied ... explicit deny`). So you cannot register a custom model from Free Edition serverless at all - the laptop (your own identity) is the only thing that can write it. At the office (a real workspace) either path works; on Free Edition, use the laptop + register_byvalue.py.

The facade was verified against BOTH states: while the endpoint was UPDATE_FAILED it returned a graceful "being updated, try again shortly" (503); once READY it returns the echo+fortune answer (200). That is the honest-classification behavior the whole project is about.

## Where it lives in the console (three different places)

- The model: Catalog -> workspace -> default -> Models -> coldstart_echo_fortune (Unity Catalog, NOT your user folder).
- The serving endpoint: Serving -> coldstart-echo-fortune (watch it build; get the URL here).
- The experiment/run: Experiments -> /Users/<your-email>/coldstart-echo-fortune.

## Measure a cold start

```bash
# Warm call - time it:
time "$DB" serving-endpoints query coldstart-echo-fortune \
  --json '{"dataframe_records":[{"prompt":"hello"}]}' -p "$PROFILE"

# Let it scale to zero (idle - minutes), then call again and time it.
# cold_latency - warm_latency = your cold-start cost. Note whether the first
# cold hit is a fast 429 ("starting"), a slow 200, or a timeout - that surface
# is what the coldstart-guard facade classifies as "warming / please hold".
```

## Cost control

Scale-to-zero means it costs nothing while idle. When done, delete it (and optionally the model):

```bash
"$DB" serving-endpoints delete coldstart-echo-fortune -p "$PROFILE"
```

## Naming

Endpoint and model names are fully custom (not reserved). To make yours unmistakable next to the system endpoints (databricks-claude-*, etc.), put your name in them, e.g. endpoint `zivkovic-coldstart-echo` and model `workspace.default.zivkovic_coldstart_echo`. Endpoints allow letters/digits/hyphens; UC model names allow letters/digits/underscores. Renaming is a redeploy: delete the endpoint, re-register under the new model name, recreate.
