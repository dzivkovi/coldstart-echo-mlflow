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

Just to confirm the model works - no account, no cloud, and **no Linux or WSL2** (that requirement was only ever about packaging for Databricks' servers, and it is already solved). Use a virtual environment so these packages stay isolated from your system Python:

```bash
python -m venv .venv          # create an isolated environment
# activate it:
#   Windows PowerShell ->  .venv\Scripts\Activate.ps1
#   macOS / Linux / WSL ->  source .venv/bin/activate
pip install -r requirements.txt
python -c "import pandas as pd; from register_byvalue import EchoFortuneModel; print(EchoFortuneModel().predict(None, pd.DataFrame([{'prompt':'hello'}])))"
```

You should see `[{'echo': 'hello', 'fortune': '...'}]`. That proves the model logic is fine. Important: this ran **entirely in your machine's memory** - it did NOT touch Databricks or any MLflow server (nothing was logged or called). You cannot reproduce a cold start locally; your machine never scales to zero. For that, deploy the endpoint below.

## Where each command runs (local vs remote)

The commands in this README go to three different places. Knowing which is which saves a lot of "wait, where did that go?":

| What you run | Runs where | Touches the Databricks cloud? |
|---|---|---|
| the local smoke test above | your machine's memory | No - nothing logged or called |
| `python register_byvalue.py` | your machine, but it **logs** to the cloud | Yes - writes the model to Databricks |
| `serving-endpoints query ...` | your machine, but it **calls** the endpoint | Yes - hits the live cloud endpoint |

"Local vs remote" in MLflow is really a question about *where you log*, and it only matters once you actually log or call something. The smoke test does neither, which is why it is safe to run with no setup.

## Deploy to Databricks (the scale-to-zero endpoint)

This is a **minimal happy path in three phases, then a wait**, not a one-liner. The full runbook - how to get the auth token, and the gotchas that bit us - is in [DEPLOYMENT.md](DEPLOYMENT.md). The short shape:

1. **Register** the model to Unity Catalog: run `register_byvalue.py` (from any machine with the Databricks CLI authenticated - any OS).
2. **Create** the endpoint: `databricks serving-endpoints create --json @endpoint_config.json`.
3. **Wait ~10 minutes**, poll until it reports `READY`, then query it.

> HAZARD: `endpoint_config.json` pins `entity_version` (currently `12`). After you register, change it to the version number your registration step printed - otherwise you deploy an old or wrong model version.

After you deploy, the pieces land in **three separate places** in the console (another common "where did it go?"):
- the **model** -> Catalog -> your catalog/schema -> Models
- the **endpoint** -> Serving (its URL, the "Query endpoint" button, and the **Logs** tab are here)
- the **run** -> Experiments

Seeing the timing logs: this model logs a `[TIMING]` line per request, but two things must be right or it shows nothing (a trap that makes people think logging is broken). The container captures the **root** logger, so your records must **propagate to root** (never `propagate=False`), and DEBUG is **not on by default** - enable it with a `LOG_LEVEL=DEBUG` environment variable on the endpoint (same screen as tracing). See the "Two gotchas" section of the performance-monitoring doc in the companion `genai-coldstart-guard` repo.

## Call it, and measure a cold start (the payoff)

Remember the point: the echo is a stand-in - the real deliverable is the **cold-start latency you measure here** and hand to the classification facade.

Model Serving endpoints have **no Swagger UI**; they take raw JSON at `/invocations` and return `predictions`. Three ways to call one:

1. **From your laptop** with the CLI (no console needed):
   ```bash
   databricks serving-endpoints query coldstart-echo-fortune \
     --json '{"dataframe_records":[{"prompt":"hello"}]}' --profile <profile>
   ```
2. **In the console**: the "Query endpoint" button on the Serving page.
3. **Raw `curl`** to the endpoint's `/invocations` URL with a bearer token - what a real application does.

To measure the cold start, time the CLI call while warm, let it sit idle (minutes) until it scales to zero, then time it again:

```bash
time databricks serving-endpoints query coldstart-echo-fortune \
  --json '{"dataframe_records":[{"prompt":"hello"}]}' --profile <profile>
```

cold latency - warm latency = your cold-start cost. Note whether the first cold hit is a fast `429` ("starting"), a slow `200`, or a timeout - that surface is exactly what the `genai-coldstart-guard` facade turns into an honest "warming / please hold" message.

## Cost

Scale-to-zero means it costs nothing while idle - that is the point. When you are finished, delete it:

```bash
databricks serving-endpoints delete coldstart-echo-fortune --profile <profile>
```
