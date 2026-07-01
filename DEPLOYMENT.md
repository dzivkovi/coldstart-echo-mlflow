# Deploying the scale-to-zero cold-start endpoint

The runbook for turning `register_byvalue.py` into a live scale-to-zero Model Serving endpoint. Every command here was actually run and works. Generic by design: no secrets, no proprietary names.

If you just want to confirm the model runs, see "Run it locally" in the [README](README.md) first - that needs no Databricks.

## Prerequisites

- The Databricks CLI, authenticated: `databricks auth login --host https://<host> --profile <profile>`.
- Python with the pinned deps: `pip install -r requirements.txt` (MLflow 2.x, pandas).
- A Unity Catalog schema you can write to (`workspace.default` is the Free Edition default; on a standard workspace use a catalog/schema you own).

## The recipe (three phases, then a ~10-minute build)

### Phase 1 - register the model

MLflow needs its own auth here (see "Why host+token" in the appendix). Pull a short-lived token and pass it explicitly:

```bash
DB=$(command -v databricks)
PROFILE=<profile>
TOKEN=$("$DB" auth token -p "$PROFILE" | grep -oE '"access_token": *"[^"]+"' | sed 's/.*: *"//;s/"//')

export DATABRICKS_HOST=https://<host>
export DATABRICKS_TOKEN=$TOKEN
unset DATABRICKS_CONFIG_PROFILE
export UC_CATALOG=workspace UC_SCHEMA=default
export MLFLOW_EXPERIMENT=/Users/<you>/coldstart-echo-fortune   # a workspace path; required

python register_byvalue.py     # prints: Registered ... version <N>
```

### Phase 2 - point the config at that version, then create the endpoint

> HAZARD: `endpoint_config.json` has `entity_version` hard-coded. Set it to the `<N>` that Phase 1 printed, or you deploy the wrong model.

```bash
# edit endpoint_config.json so "entity_version" matches <N>, then:
MSYS_NO_PATHCONV=1 "$DB" serving-endpoints create --json @endpoint_config.json -p "$PROFILE"
#   ^ this call blocks waiting for readiness and may "time out" in a shell after ~2 min.
#     That is fine - the build continues. Do NOT re-run it; poll instead (Phase 3).
```

(`MSYS_NO_PATHCONV=1` is only needed on Windows Git Bash, which otherwise mangles the `/serving-endpoints/...` path.)

### Phase 3 - wait until READY

Custom model serving builds a container and loads the model - several minutes.

```bash
until "$DB" serving-endpoints get coldstart-echo-fortune -p "$PROFILE" | grep -q '"ready": "READY"'; do
  echo "building..."; sleep 20
done
"$DB" serving-endpoints query coldstart-echo-fortune \
  --json '{"dataframe_records":[{"prompt":"hi"}]}' -p "$PROFILE"
# -> {"predictions":[{"echo":"hi","fortune":"..."}]}
```

## Redeploy a new version (after you edit the model)

Do NOT delete and recreate the endpoint. Register a new version and **swap** the endpoint to it with `update-config`: the endpoint keeps its name and URL, stays live on the old version until the new one is READY (no downtime), and keeps scale-to-zero.

```bash
# 1. register the new version
python register_byvalue.py            # prints "Registered ... version <N>"

# 2. read the newest version number and point the endpoint at it
V=$(databricks model-versions list workspace.default.coldstart_echo_fortune -p <profile> \
     | grep -oE '"version": *[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1)
databricks serving-endpoints update-config coldstart-echo-fortune -p <profile> --json \
  "{\"served_entities\":[{\"entity_name\":\"workspace.default.coldstart_echo_fortune\",\"entity_version\":\"$V\",\"workload_size\":\"Small\",\"scale_to_zero_enabled\":true}]}"

# 3. wait until READY, then query
until databricks serving-endpoints get coldstart-echo-fortune -p <profile> | grep -q '"ready": "READY"'; do
  echo building...; sleep 20
done
databricks serving-endpoints query coldstart-echo-fortune --json '{"dataframe_records":[{"prompt":"hello"}]}' -p <profile>
```

A code-only change may cut over in seconds (cached container); a dependency change is a full ~10-min rebuild. Changing an endpoint environment variable (e.g. `LOG_LEVEL=DEBUG`) is also a config change and redeploys the same way. Keep `entity_version` in `endpoint_config.json` in sync with `<N>` (or just use the `update-config` command above, which takes it inline).

## Where it lives in the console (three separate places)

- The model: **Catalog -> workspace -> default -> Models -> coldstart_echo_fortune** (Unity Catalog, not your user folder).
- The serving endpoint: **Serving -> coldstart-echo-fortune** (watch it build; the URL is here).
- The run/experiment: **Experiments -> /Users/<you>/coldstart-echo-fortune**.

## Reading the endpoint logs (where [TIMING] / [COLDSTART] show up)

Easiest: the **Logs** tab on the Serving page (Serving -> coldstart-echo-fortune -> Logs). Or the CLI:

```bash
# the served-model name is <model>_name-<version>, e.g. coldstart_echo_fortune-15;
# read it from the endpoint's served_entities[].name:
SM=$(databricks serving-endpoints get coldstart-echo-fortune -p <profile> | grep -oE 'coldstart_echo_fortune-[0-9]+' | head -1)
databricks serving-endpoints logs coldstart-echo-fortune "$SM" -p <profile> | grep -E 'TIMING|COLDSTART'
```

Remember: `[COLDSTART] worker_boot` is WARNING so it always shows; the per-request `[TIMING]` and `[COLDSTART] request` lines are DEBUG, so they only appear when `LOG_LEVEL=DEBUG` is set as an endpoint environment variable (which is a config change - it redeploys).

## Cost

Scale-to-zero costs nothing while idle. Delete when finished:

```bash
"$DB" serving-endpoints delete coldstart-echo-fortune -p "$PROFILE"
```

## Naming

Endpoint and model names are fully custom (not reserved words). To make yours obvious next to the built-in system endpoints (`databricks-claude-*`, etc.), put a personal prefix on them, e.g. endpoint `<name>-coldstart-echo` and model `workspace.default.<name>_coldstart_echo`. Endpoints allow letters/digits/hyphens; Unity Catalog model names allow letters/digits/underscores. Renaming is a redeploy (delete, re-register under the new name, recreate).

---

## Appendix: why it is done this way (so you do not repeat our detours)

Two auth details that are easy to trip on:

- **Why host+token, not the CLI profile (Phase 1).** MLflow could not use the OAuth CLI profile directly - it failed to build the SDK client and fell back to no credentials (`401: Credential was not sent`). Passing `DATABRICKS_HOST` + `DATABRICKS_TOKEN` (and unsetting `DATABRICKS_CONFIG_PROFILE`) is the clean fix without creating a long-lived personal token. The token is a ~1-hour bearer, plenty for a registration.
- **Why `MLFLOW_EXPERIMENT`.** On Databricks tracking, `mlflow.start_run()` fails with `RESOURCE_DOES_NOT_EXIST: experiment ID None` unless a workspace experiment is set first. `register_byvalue.py` sets it from that env var.

Two dead ends we ruled out (each cost a build cycle):

- **Logging a code file instead of by value.** MLflow's "models from code" recorded an absolute source path (a `C:\...` path when registering from Windows) that the Linux serving container cannot resolve, so the model failed to load (endpoint reached `UPDATE_FAILED`, "Model server failed to load the model"). Registering **by value** - logging the class instance, as `register_byvalue.py` does - removes the source path entirely and fixes this. It was the path, not the Python version (3.12 serves fine).
- **Registering from Free Edition serverless.** Running the registration on Free Edition serverless compute (a job, or an interactive notebook) is explicitly denied write access to the model storage (`s3:PutObject ... AccessDenied ... explicit deny`). So on Free Edition you must register from a machine that authenticates as **you** (your own CLI session), not from serverless. On a standard workspace this restriction does not apply - which is why `databricks_notebook.py` is a standard-workspace-only path.

The facade in `genai-coldstart-guard` was verified against both states of this endpoint: while it was building/failed it returned a graceful "the service is being updated, try again shortly" (503); once READY it returns the echo+fortune answer (200). That honest classification is the whole point of the companion project.
