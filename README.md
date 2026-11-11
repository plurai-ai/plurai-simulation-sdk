# Plurai Simulation SDK

The client runtime of the [Plurai](https://plurai.ai) simulation product. It runs simulated users against your AI
agent, mocks the tools your agent calls, and sends the conversations to the Plurai simulation backend to be scored on
your policies.

You normally meet this package through the Plurai plugin for Claude Code, which installs it, generates a `simulate.py`
for your agent and drives the commands below. Everything the plugin does goes through the interfaces documented here, so
you can also run them yourself.

- Python 3.12 or 3.13 on macOS, Linux or Windows. On Windows the saved login lives in your user profile folder, and a coding agent that is driven through its command-line tool (`claude_agent_sdk`) is not supported on Windows; run it on macOS, Linux or WSL.
- The distribution is `plurai-simulation-sdk`. The import name and the command lines are `plurai`.

## Install

```bash
pip install plurai-simulation-sdk
```

The base install is small: the mock engine, an HTTP client and the command lines. Everything heavier is an extra, so you
install only what you use.

| Extra | Install | Adds |
| --- | --- | --- |
| `readers` | `pip install "plurai-simulation-sdk[readers]"` | the document readers a generated `simulate.py` previews agent output with (`.docx`, `.pptx`, `.xlsx`, `.pdf`) |
| `extract` | `pip install "plurai-simulation-sdk[extract]"` | client-side corpus extraction for `plurai ingest-corpus` |
| `extract-pdf`, `extract-http`, `extract-vertex`, `extract-bigquery`, `extract-redshift`, `extract-azure-sql`, `extract-databricks`, `extract-snowflake`, `extract-bedrock` | `pip install "plurai-simulation-sdk[extract-bigquery]"` | the driver for one corpus source kind |
| `anonymize` | `pip install "plurai-simulation-sdk[anonymize]"` | local PII scrubbing before upload (also needs a spaCy English model) |
| `anonymize-binary` | `pip install "plurai-simulation-sdk[anonymize-binary]"` | the same for images and PDFs (also needs the Tesseract and Poppler programs) |
| `langsmith`, `langfuse`, `arize`, `braintrust` | `pip install "plurai-simulation-sdk[langfuse]"` | the matching observability sink |

`plurai login`, `plurai --version` and `plurai capabilities` need none of the extras.

## Quickstart

1. **Sign up and log in once.** The command opens your browser on the Plurai sign-in page, which also signs you up. Accept the
   consent screen and pick your organization; the command then creates its own API key and saves it. The key is never
   shown, and the sign-in url is printed too, for a machine where no browser opens.

   ```bash
   plurai login
   plurai login --status
   ```

   On a machine with no browser, or for a backend the login does not know, export `PLURAI_URL` and `PLURAI_TOKEN`
   (a key made in the dashboard) instead.

   `plurai logout` revokes the saved key at the backend and removes the credentials file. When the backend cannot be
   reached it keeps the file and exits 75, so you can run it again; when no retry can help it removes the file, names
   the key that still works and points to the dashboard (exit 77).

2. **Check who you are.** `plurai login --status` names the backend, the organization, the user, and the key's name and
   expiry. It never prints the key.

3. **Run a first simulation.** With a `simulate.py` for your agent and a scenario set on the backend:

   ```bash
   plurai-run-simulation \
     --scenario-set-id <scenario set id> \
     --agent-module-path plurai_sim/simulate.py \
     --out runs/first
   ```

   Each scenario is a conversation between a simulated user and your agent. The runner writes one folder per scenario
   under `--out`, then `run_summary.json` and prints the verdicts. The exit code says how the run ended; the table is under
   Exit codes below.

If you have the Claude Code plugin, its `/simulate` command does steps 1 to 3 for you.

## Authentication

Every command resolves its backend the same way. There is no default backend, and no `--server-url` or `--token` flag:

1. `PLURAI_URL` together with `PLURAI_TOKEN` from the environment, else
2. the pair saved by `plurai login` in `~/.plurai/credentials.json`.

The pair is taken whole from one source. A url from one source with the token from the other is an error, because a
token must only reach the backend it was issued for. `PLURAI_URL` must be `https`; plain `http` is accepted only for a
loopback address, which needs no token. A `.env` file is not a source. The token is sent only to the resolved origin
(scheme, host and port), so a mock url on another host or on another local port never receives it. With nothing
configured, a command prints how to sign up and log in instead of trying to connect.

`plurai login` signs in to production, or to the backend `PLURAI_URL`
names when that is one the SDK knows; any other host is refused. When only `PLURAI_URL` is set, the login saves its key
but every other command refuses that configuration until `PLURAI_URL` is unset or `PLURAI_TOKEN` is exported too. With
`PLURAI_TOKEN` set it refuses, because the environment pair is what every command uses, and with a usable saved key for the same backend it does nothing. A login that overwrites a saved key revokes it with its own key once the
new key is saved, and warns when it cannot; keys made in the dashboard are never touched. It never keeps the sign-in's own
tokens: the refresh token is revoked once the key is saved, or when the key could not be created. CI and other machines without a browser use `PLURAI_URL` and `PLURAI_TOKEN`, or a
credentials file written elsewhere. The credentials file is readable by anything that runs as you, a shell an agent runs
included. Revoke a key in the dashboard; deleting the file only forgets it.

| Credentials file | |
| --- | --- |
| Path | `~/.plurai/credentials.json`, in the home directory the account database names |
| Content | `{"url": "<url>", "token": "<token>"}`, and after a browser login an `account` object (organization, user, key name and id, expiry) that holds nothing secret |
| Safety | directory mode 0700, file mode 0600, both owned by you, neither a symlink; written atomically |

The agent under test does not inherit your shell's environment. It starts with what a process needs to run (`PATH`,
`HOME`, locale, TLS and proxy settings, the sign-in variables of its Claude Code CLI), the run's resolved `PLURAI_URL`
and `PLURAI_TOKEN`, and the variables you list under `agent.env` in `plurai.config.yaml`. The token is redacted from
every artifact, log line and observability payload the SDK writes.

## Grouping a command's jobs

`plurai-run-simulation --command-run-id <id>`, or `PLURAI_COMMAND_RUN_ID` in the environment (the flag wins), labels the
run's requests with an id so the backend can list every job one command started together. The id is 1 to 128 visible
characters with no spaces; anything else exits 64. It is a grouping label only: a run with no id works the same. The Claude
Code plugin sets it for the commands it starts.

## What it uploads

Two things leave your machine. `plurai-run-simulation` sends each conversation to the Plurai backend while it runs: the
simulated user's messages, your agent's replies, its tool calls and what they returned, so that the backend can play the
user and score the result. The `plurai ingest-*` commands upload customer data you point them at, and `plurai stash-doc`
uploads one document you name and prints its size on stderr.

- **Before an ingest upload**, a disclosure is printed on stderr, and the upload is recorded in the append-only
  `.plurai/uploads/audit.log`. For `ingest-corpus`, `ingest-traces` and `ingest-descriptions` the disclosure lists every
  part with its size, its hash and its anonymization posture, and is also saved in `.plurai/uploads/`. It links the
  Terms of Service and the Privacy Policy that govern the upload.
- **The default posture is `raw`**: data is uploaded as it is and processed under your agreement with Plurai. Set
  `anonymize: true` in `plurai.config.yaml` (needs the `anonymize` extra) and the SDK scrubs personal data locally before
  upload, replacing it with a one-way keyed hash. The scrub fails closed: a missing package or model stops the run
  rather than uploading unscrubbed data. The key stays in a gitignored `.plurai/anon.key` and is never uploaded.
- **The posture governs only the data uploaded while building the environment.** A simulation run's own artifacts
  (`ingest-run`, `ingest-run-report`) always upload as they are, whatever `anonymize:` says.
- **An environment never mixes postures.** A build upload that would add a differently postured upload to an
  environment that already has one is refused.
- **An archive over the backend's size cap never leaves the machine.** The SDK checks the cap first and names the largest
  files.
- **The scan contract runs on your machine.** `plurai-scan-contract check` reads your corpus locally and writes its
  report locally. It reaches your running agent only when you pass both `--probe-consent` and `--agent-command`, and then
  it sends the probes to your own command, never to a Plurai service.
- **No telemetry.** The SDK sends nothing about your use of it. A support engineer sees only what you paste.

Observability sinks (LangSmith, Langfuse, Arize, Braintrust) are separate: when you declare one, the run records to your
own account on that platform with your own keys.

## Sandbox profile for Claude-based agents

When the agent under test is built on `claude_agent_sdk`, the SDK confines each scenario with `plurai.confine`. The
agent's shell runs in a sandbox, its built-in tools are denied unless the harness grants them, only the MCP servers the
agent declares are loaded, and every file the agent reads is judged against a read scope. The record of what was applied
and what was observed travels with the run.

The default profile is permissive. It keeps what protects the run and your machine, and leaves the agent's shell its
network and its toolchain: the harness and the expected answers, other scenarios and your credentials cannot be read or
reached from the agent's shell.

Set `PLURAI_SANDBOX=strict` for the strict profile. It closes the shell's sandbox further: no network, no extra system
services, nothing writable outside the scenario's own folders, and no toolchain reads under your home directory. Use it
when the agent does not need the network to do its work, or when you want the tightest isolation a run can have.

```bash
PLURAI_SANDBOX=strict plurai-run-simulation --scenario-set-id <id> --agent-module-path plurai_sim/simulate.py --out runs/strict
```

Any other value of `PLURAI_SANDBOX`, or none, selects the permissive profile.

## Writing `simulate.py`

`import plurai` is the supported surface for a `simulate.py`: the names in `plurai.__all__` and nothing else.

```python
import plurai

@plurai.mock('get_order')
def get_order(order_id: str) -> dict:
    return plurai.world_state.get_entity(order_id)
```

| Area | Names |
| --- | --- |
| Mocks and context | `mock`, `observe`, `MockToolNode`, `mock_wrap`, `apply_mocks`, `mock_result`, `MockToolError`, `SimulationContext`, `get_simulation_context`, `world_state`, `world_state_snapshot` |
| Session | `configure`, `start_session`, `get_full_state`, `get_events`, `active_session_id`, `mock_service_url`, `MockService`, `MockServiceClient` |
| Faults | `HarnessFault` (the harness refuses a path it does not cover; the scenario is recorded as not valid, not scored against the agent) and `AgentFault` (the agent failed where no Python frame can show it) |
| Host effects | `cli_hooks`, `confine`, `host_effects_granted`, `host_read_scope` |
| Files and rows | `file_row`, `store_record`, `model_input_context`, `ToolDescriptor` |
| The `code` action channel | `code_intercept_call`, `install_code_hook`, `install_code_hook_from_env`, `uninstall_code_hook`, `sitecustomize_source`, `write_sitecustomize`, `watch_interception`, `build_code_surface`, `emit_code_surface`, `CodeSurfaceGenerationError` |

The REST client types (`Orchestrator`, `OrchestratorClient` and their errors) are in `plurai.orchestrator`.

### The 1.0 contract

The following stay compatible within a major version:

- the names in `plurai.__all__` and in `plurai.observability.__all__`;
- the flags of the three command lines, `plurai`, `plurai-run-simulation` and `plurai-scan-contract`;
- the exit codes below;
- the credentials file above;
- the `PLURAI_*` environment variable names;
- the set of keys `plurai capabilities` prints.

Everything else is an implementation detail and may change in any release. The files the plugin reads from a run
(`run_summary.json`, `.last_run`) may change without a major release.

## Exit codes

All three command lines exit with the same table. A code means one thing everywhere, and a command uses only the rows
that apply to it. 2 is never returned: it is the status of a bad command line in most tools, and here that is 64.

| Code | Name | Meaning |
| --- | --- | --- |
| 0 | `OK` | done; everything checked held |
| 1 | `POLICIES_FAILED` | the run completed and at least one scenario failed a policy |
| 10 | `UPGRADE_REQUIRED` | the backend refuses this client version (HTTP 426); the command prints `pip install -U 'plurai-simulation-sdk>=<minimum>'`; under the Plurai CI action it asks for a newer action release instead |
| 11 | `MOCK_UNANSWERED` | a mocked tool call could not be answered, so the run was stopped |
| 12 | `ROUTE_MISSING` | the backend predates the route the command calls |
| 13 | `JOB_FAILED` | a backend job ended in failure |
| 14 | `STILL_RUNNING` | `plurai job wait` stopped waiting with the job still running; run it again |
| 15 | `NOT_FOUND` | the job, run or artifact does not exist |
| 16 | `NOT_READY` | the artifact is still a draft |
| 17 | `BACKEND_REFUSED` | the backend refused what was sent; sending it again fails the same way |
| 18 | `SCAN_HELD` | `plurai-scan-contract` verified the sources and a finding holds the flow |
| 19 | `REQUIREMENT_GAP` | `python -m plurai.host_toolchain` measured a gap on the host |
| 20 | `NOT_MEASURED` | a command that checks policies could not measure them: an infrastructure error, a suite fault or no credit; the report names the cause |
| 64 | `USAGE` | bad command line |
| 65 | `REFUSED` | declined to proceed: a gate holds the run, or the input or platform is unusable |
| 66 | `INPUT_UNREADABLE` | a local input file is missing, empty or unreadable |
| 69 | `TRANSPORT` | the backend or the mock was unreachable or gave an unusable answer |
| 70 | `INTERNAL_FAULT` | a defect in the SDK; nothing was concluded |
| 75 | `TEMPFAIL` | temporary: the sign-in timed out or a server asked to retry later (HTTP 429); run the command again later |
| 77 | `CREDENTIALS` | no usable credential, or the backend rejected the token |
| 130 | `INTERRUPTED` | the user interrupted the command (Ctrl-C); a run keeps the per-scenario artifacts it had written and writes no `run_summary.json` |

The names are constants in `plurai.exit_codes`. A run that completed with failing policies exits 1 unless something more
specific applies; a run summary's `exit_code` field carries the same number.

A run stops with 11 when a hosted mock cannot answer in a way no retry changes: the token is rejected (`auth`), the mock
env is not deployed (`not_deployed`) or the url is not a mock (`not_a_mock`). Nothing new starts, the running scenarios
are cancelled, no score is reported, and stderr names what to change. `run_summary.json` then carries an `aborted` block
(`reason`, `cause`, `scenario`, `tool`, `status_code`, `remedy`) and no policy verdict. A dropped connection is retried
and does not stop the run.

A run stops with 20 when the backend refuses work for credit, in any scenario and at any step, including a refusal that
the gateway in front of the backend answers with a 402 whose body names the kind. Nothing new starts and the scenarios
still in their dialogs are cancelled. A scenario that is already waiting for its finalize job finishes and keeps its
verdict. The scenarios that finished keep their artifacts and verdicts, and the others are recorded as `incomplete` and
listed in `not_run_scenarios`, never as failed. A stopped run does not write `.last_run`. stderr prints one sentence for
the kind of refusal (`insufficient-credit`, `new-run-refused`, `free-tier-ceiling`, `credit-unavailable` or
`credit-refused`). The three kinds a payment fixes name the billing page, or "the Plurai dashboard" when the environment has none;
`plurai capabilities` reports the page as `billing_url`. `credit-refused` only asks the user to try
again later and names no link, contact or error id. `credit-unavailable` is a fault on Plurai's side, so it prints the
contact hand-off: one line with the support email and the Discord server, and no mention of credit.
`run_summary.json` carries `aborted: {reason: credit, code, resets_at?, scenario?, remedy}`; `scenario` names the
scenario that met the refusal, and is absent when the refusal came before any scenario began.
A run also stops with 20, the same way, when a hosted mock's answer carries the backend's typed capacity block
(`code: capacity_exhausted`, `resets_at`, `reset_source`), on the mock's fault or on its answer to the call. Every call
is refused the same way meanwhile, so the run is never played on into refusals. stderr prints
one fixed sentence at the end of the run ("Plurai is in high demand right now ... try again in about 20 minutes"),
and never reads `resets_at`; scenario lines say only that the backend could not take the run and the
scenario was not run. `run_summary.json` carries
`aborted: {reason: capacity, code: capacity_exhausted, resets_at?, reset_source?, scenario, remedy}`.
A run stops with 20, the same way, when the backend stays unavailable. An outage is a 503 or a failed connection on the
create, or a 502, 503 or 504 or a failed connection on a turn, that outlasts the retry budget (about 60 seconds, so a
backend upgrade passes unseen; `--retries` sets the count). It is recovered first: the scenario that met it waits for the
backend to answer `/health` again (up to 5 minutes) and is then re-run once on a fresh thread, and the run goes on. The run
stops only when the backend stays down past that wait, or the re-run meets the outage again. A finalize poll that meets an
outage keeps polling its job by id for its own time budget, since the job's judges already ran. A hosted mock's 503, 502
(the proxy could not connect to the mock) or failed connection that outlasts the budget, or a hosted mock's answer carrying
the backend's typed `unavailable` block (read by where it sits, beside where `capacity` sits; its `code` is recorded and
never read, so a new code needs no client release), fails that one scenario, which is re-run once like any transient mock
fault; the run does not stop on it. A typed `unavailable` block the backend sends as its own answer stops the run at once.
A request the backend may already have taken (the create, a mock call) is never re-sent after a 502, a 504 or a broken
connection; that one failure fails its scenario and is no outage. A stopped run prints one fixed
sentence at the end of the run ("Plurai is unavailable right now. Please try again in a few minutes."); scenario lines
say only that the backend could not take the run and the scenario was not run (or not completed, when it had begun),
and no line carries a status code or an id. The stop tries to close every open thread (`/fail`) and to cancel its
finalize jobs, with one short try each (a few seconds), so a thread left open does not refuse the organization's next
run; a backend that does not answer in that time keeps the thread until its idle timeout. A process the agent spawns for the `code` action channel (a Node CLI, a Python
skill) waits out the same budget, but it has no channel to the run: a spent budget there ends as the tool error the agent reads,
and the run does not stop on it. A stopped run does not write `.last_run`. `run_summary.json` carries
`aborted: {reason: unavailable, cause: transport | status | backend, code?, scenario, remedy}`: `transport` is a connection that
failed or broke, `status` a 502, 503 or 504, `backend` the backend's own typed block, which stops the run at once. `code` is the typed
block's own code when it named one, and is absent for `transport` and `status`. The 429
throttle keeps its own retries.
One refusal is not final: a `new-run-refused` answer to a conversation that has not started yet (the create, or
its first turn) makes that scenario wait while the run has conversations of its own that the gateway admitted (their
first turn landed). It keeps its concurrency slot, and one waiting scenario tries again when one of the run's own
conversations ends; when that try is admitted the next waiting scenario tries at once, and a refused try waits a fixed
interval before the next one. The gateway frees an ended conversation's room a few minutes after the end, so a scenario
can wait a little longer than its sibling took. A refusal that arrives while siblings are still starting says nothing
until one is admitted (then it waits) or none is left starting (then the run stops). When nothing of the run's own was
admitted or is held and a start is still refused, the run stops exactly as above, with no wait reported. A refusal of any
other kind, or a `new-run-refused` on a later turn, a finalize or
a judge replay, stops the run at once. `run_summary.json` carries `credit_waits: {scenarios, refusals, seconds}`
when a start waited, and no such key otherwise. Nothing else retries a credit refusal. With `--trials N` every trial keeps its own summary, no cross-trial aggregate is
written, and the exit is 20 whatever the finished trials returned; `$OUT/run_summary.json` carries the `aborted` block and
exit code 20 even when trial 1 finished before the refusal.

A hosted mock that refuses a model call for credit stops the run the same way, whether it answers with an in-band fault
(`cause: credit` and the credit block beside it) or the call comes back as a 402. The sentence, the summary and the exit
code are the ones above. `plurai job wait` on a job whose `terminal_reason` is `credit`, or whose poll the gateway refuses
with a 402 that names a credit kind, exits 13 (`JOB_FAILED`) at once and prints the same sentence for the kind, without the
job's generic error text or error id. For the refused poll it says the status could not be read, not that the job failed:
the job may have finished. `plurai job status` exits 13 with the sentence for the same 402. `plurai fetch`,
`plurai backend identity`, `plurai stash-doc` and every `plurai ingest-*` command exit 13 with the sentence for a 402 that
names a credit kind, on the storage upload or the stash.

This SDK and a backend that reports credit refusals go together: `terminal_reason` is a closed set, so an SDK from before
credit refusals fails on a job that ends `terminal_reason: credit` instead of reading it. Update the SDK with the backend.

## Commands

| Command | Does |
| --- | --- |
| `plurai-run-simulation` | runs a scenario set against your agent and writes the per-scenario artifacts |
| `plurai login`, `plurai login --status` | saves, or reports, the backend login |
| `plurai logout` | revokes the saved key and removes the credentials file |
| `plurai diagnose` | prints a bundle to paste into a support issue |
| `plurai capabilities`, `plurai --version` | print the version, the commands and the feature keys of this build |
| `plurai ingest-corpus`, `ingest-traces`, `ingest-doc-examples`, `ingest-bundle`, `ingest-descriptions` | extract, optionally anonymize, and upload customer data, after the disclosure |
| `plurai ingest-run`, `ingest-run-report` | upload a simulation run's artifacts as-is (never anonymized, whatever `anonymize:` says), after the disclosure |
| `plurai convert-trace` | turns a production trace export into a scenario that resumes from it |
| `plurai stash-doc FILE LABEL` | uploads one generation input (a PRD, a variables file) and prints its id on stdout |
| `plurai job status ID`, `plurai job wait ID` | read a backend job, or block until it ends |
| `plurai backend identity`, `plurai backend probe` | ask which deployment answers, or whether it answers |
| `plurai fetch policies`, `plurai fetch artifact` | download backend-owned files |
| `plurai correct-scores` | overwrite corrected scores on your observability platform |
| `plurai-scan-contract check`, `recheck` | verify a corpus against the facts the agent's description claims, on your machine |

Run any command with `--help` for its flags. A command writes only what a caller parses to stdout and everything else to
stderr under its own prefix.

## Observability sinks

Declare at most one sink with `<SINK>__ENABLED=true` and its credentials, for example `LANGFUSE__PUBLIC_KEY` and
`LANGFUSE__SECRET_KEY`. Whether a run records to it is a separate declaration, never a discovery: pass `--observability`,
set `PLURAI_OBSERVABILITY=recording` in the real environment, or write `recording` into a `.observability` file beside
`simulate.py`. The default is isolated: nothing is recorded. A missing package or a sink failure never breaks a run.

| Sink | Install | Credentials |
| --- | --- | --- |
| LangSmith | `pip install "plurai-simulation-sdk[langsmith]"` | `LANGSMITH__API_KEY` |
| Langfuse | `pip install "plurai-simulation-sdk[langfuse]"` | `LANGFUSE__PUBLIC_KEY`, `LANGFUSE__SECRET_KEY` |
| Arize | `pip install "plurai-simulation-sdk[arize]"` | `ARIZE__API_KEY`, `ARIZE__SPACE_ID` |
| Braintrust | `pip install "plurai-simulation-sdk[braintrust]"` | `BRAINTRUST__API_KEY` |

## Versions and updates

The Claude Code plugin pins the exact version of this SDK it was tested with. Update the plugin to move to a new SDK
version, and do not upgrade the SDK on its own unless a command asks you to.

## Getting help

Open an issue on [the project's GitHub page](https://github.com/plurai-ai/plurai-simulation-sdk/issues) with one of the
forms (bug, install problem, question) and paste the output of:

```bash
plurai diagnose
```

The bundle holds the SDK version, the installed extras, the Python and OS versions, whether a credential and a backend
url are present, the backend's origin but never the token, the latest run's name and error ids, and the references other
Plurai clients recorded locally. It holds no token and no file from your project.

If the install itself failed, `plurai diagnose` cannot run. Paste the output of `pip install -v plurai-simulation-sdk`
(remove a password from any index url first), `python --version`, `pip --version`, and your operating system. Never paste
`~/.plurai/credentials.json` or a token.

See the [changelog](https://github.com/plurai-ai/plurai-simulation-sdk/blob/main/CHANGELOG.md) for what changed in each
release.
