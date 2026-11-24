"""``plurai ci run``: the orchestration of a CI simulation, from offline rules to the exit code.

Each step reads its inputs, calls the layer that owns the rule and either goes on or ends the run with a :class:`Stopped`.
The rules live elsewhere: the offline rules in :mod:`plurai.ci.config`, the online rules in :mod:`plurai.ci.online`, the
verdict layer in :mod:`plurai.ci.overrides`, :mod:`plurai.ci.retry`, :mod:`plurai.ci.quarantine` and :mod:`plurai.ci.verdict`,
and the renderings in :mod:`plurai.report`. This module writes the files of ``PLURAI_CI_OUT_DIR`` and nothing else.

The run loop keeps its state in a :class:`RunState`: the log of every thread id opened, the output directory of each pass and a
snapshot of the run so far, which is what the signal handler (:mod:`plurai.ci.interrupt`) needs to cancel the open threads
(:mod:`plurai.ci.cancel`) and write a partial ``run.json``.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import signal
import sys
import tempfile
import time
import traceback
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Final

from plurai import __version__, client_handshake, exit_codes, run_origin
from plurai import _endpoints as endpoints
from plurai._endpoints import Endpoint
from plurai.ci import run_doc
from plurai.ci.cancel import Cancels, Released
from plurai.ci.gitrepo import head_commit
from plurai import atomic_write
from plurai.ci.backend import RESUME_SECONDS, Backend, BackendUnreachable, TokenRefused, TokenRejected, UpgradeRefused
from plurai.ci.config import REGRESSION_PREFIX, Offline, Refusal, check_offline, today_utc
from plurai.ci.interrupt import CHILD_GRACE_SECONDS, WORK_DEADLINE_SECONDS, Interrupted, SignalGuard, Signalled, stop_children
from plurai.ci.mock_wait import wait_for_mock
from plurai.ci.online import StopRun, Suite, check_credits, check_ids, check_smoke_names, stop_for
from plurai.ci.overrides import OverridePlan, apply_overrides, plan_overrides
from plurai.ci.quarantine import quarantine_stop
from plurai.ci.retry import cap_reason, merge_retry, retry_note, retry_stop, select_retries
from plurai.ci.runner import PassPlan, agent_version, run_pass
from plurai.ci.suite import BaseUnreadable, Diffed, SuiteNames, base_ids_at, diff_against, ids_map, suite_names
from plurai.ci.verdict import verdict
from plurai.orchestrator import thread_log
from plurai.report import (
    RUN_SCHEMA,
    ReportOptions,
    Run,
    RunJsonError,
    Stopped,
    SuiteDiff,
    Visibility,
    is_plain_web_address,
    level_of,
    read_run,
    render_comment,
    render_ids_only,
    render_junit,
    render_report,
    render_summary,
    resolve_include_content,
    run_json_for_upload,
    tags_of,
    tally,
)

SELECTIONS: Final = ('smoke', 'full')
VISIBILITIES: Final = ('public', 'private', 'internal')
#: The environment variable the simulation action sets to its own version; the command runs without it outside the action.
ACTION_VERSION_ENV: Final = 'PLURAI_ACTION_VERSION'

#: The refusal behind the runner's upgrade-required exit. The runner prints its own text and returns the exit code, so the
#: answer's reason and floor are not available here.
_RUNNER_REFUSAL: Final = client_handshake.UpgradeRefusal(reason=None, floor=None)

_REPO: Final = re.compile(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+')
_TRANSCRIPTS: Final = 'transcripts'


@dataclass(frozen=True)
class RunArgs:
    """The command line of ``plurai ci run``, as strings: the command validates them as its first step."""

    repo: str
    selection: str
    base: str | None = None
    main_run: str | None = None
    visibility: str | None = None
    include_content: str | None = None
    run_url: str | None = None


@dataclass
class RunState:
    """What a run in progress knows that a caller stopping it needs: every thread id it opened (cancelling one that already
    finished is harmless) and the output directory of each pass of the runner."""

    threads: thread_log.ThreadLog = field(default_factory=thread_log.ThreadLog)
    pass_dirs: list[Path] = field(default_factory=list)
    #: The directory the passes write into, removed when the run ends.
    scratch: Path | None = None
    #: The run so far as ``(run, run.json document)`` for a cancel, or ``None`` when nothing was judged yet. Set while the passes run.
    snapshot: Callable[[], tuple[Run, dict[str, Any]] | None] | None = None
    #: Names this invocation's passes, so no two share a run name.
    token: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


class _Finished(Exception):  # noqa: N818
    """The run ended at a step: its report and exit code are written."""

    def __init__(self, code: int) -> None:
        super().__init__(str(code))
        self.code = code


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temporary file in the same directory and a rename, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write.write_atomic(path, text)


def _plural_threads(count: int) -> str:
    return f'{count} open thread{"" if count == 1 else "s"}'


#: What a run may write into the output directory and a later run into the same directory must not inherit.
_STALE: Final = ('run.json', 'junit.xml', 'suite-diff.md', 'suite-diff-comment.md')


def clear_stale_outputs(out_dir: Path) -> None:
    """Remove the files an earlier run left that this run may not write: a refusal has no ``run.json``, and a run whose ids did
    not move has no suite diff."""
    for name in _STALE:
        (out_dir / name).unlink(missing_ok=True)
    shutil.rmtree(out_dir / _TRANSCRIPTS, ignore_errors=True)


def _bool_flag(value: str | None) -> bool | None:
    return None if value is None else value == 'true'


@dataclass
class _Run:
    """One invocation. Methods are the steps, in order; a step that stops the run calls :meth:`end`."""

    args: RunArgs
    env: Mapping[str, str]
    workdir: Path
    out_dir: Path
    today: date
    sleep: Callable[[float], None]
    state: RunState
    offline: Offline | None = None
    suite: Suite | None = None
    diff: SuiteDiff | None = None
    notes: list[str] = field(default_factory=list)
    token_expires: date | None = None
    credit_state: str | None = None
    names: SuiteNames | None = None
    main: Run | None = None
    main_error: str | None = None
    #: The committed ids at ``--base``, when they could be read: they differing from main's last run says main's suite moved.
    base_ids: dict[str, str] | None = None
    endpoint: Endpoint | None = None
    #: Called when the run starts writing its final reports: a signal from then on has nothing to add.
    disarm: Callable[[], None] = field(default=lambda: None)

    # --- shared pieces --------------------------------------------------------------------------------------

    @property
    def include_content(self) -> bool | None:
        return _bool_flag(self.args.include_content)

    @property
    def visibility(self) -> Visibility | None:
        value = self.args.visibility
        return value if value in ('public', 'private', 'internal') else None

    @property
    def content_included(self) -> bool:
        return resolve_include_content(self.include_content, self.visibility)

    def identity(self, *, with_ids: bool) -> dict[str, Any]:
        doc: dict[str, Any] = {
            'repo': self.args.repo,
            'selection': self.args.selection,
            'versions': {'sdk': __version__},
            'finished_on': self.today.isoformat(),
        }
        if action := self.env.get(ACTION_VERSION_ENV, '').strip():
            doc['versions']['action'] = action
        if commit := head_commit(self.workdir):
            doc['commit'] = commit
        if with_ids and self.offline is not None:
            doc.update(self.offline.ids.as_run_fields())
            doc['env_alias'] = self.offline.config.env
        if with_ids and self.suite is not None:
            doc['suite_scenarios'] = len(self.suite.scenario_names)
            doc['pinned_digests'] = self.suite.digests()
        if with_ids and self.names is not None:
            if self.names.scenario_titles:
                doc['scenario_titles'] = self.names.scenario_titles
            if self.names.policy_descriptions:
                doc['policy_descriptions'] = self.names.policy_descriptions
        if self.credit_state is not None:
            doc['credit_preflight'] = self.credit_state
        if self.token_expires is not None:
            doc['token_expires'] = self.token_expires.isoformat()
            doc['token_days_left'] = (self.token_expires - self.today).days
        return doc

    @property
    def run_url(self) -> str | None:
        """The ``--run-url`` when it is usable: the report that says it is refused must not carry it."""
        url = self.args.run_url
        return url if url is not None and is_plain_web_address(url) else None

    def options(self, *, stopped: Stopped | None = None, measured: bool = True) -> ReportOptions:
        artifacts = ['run.json', 'report.md', 'junit.xml'] if measured else ['report.md']
        if self.diff is not None and self.diff.moved:
            artifacts.append('suite-diff.md')
        return ReportOptions(
            main=self.main,
            include_content=self.include_content,
            run_url=self.run_url,
            visibility=self.visibility,
            main_error=self.main_error,
            base_ids=self.base_ids,
            notes=tuple(self.notes),
            artifacts=tuple(artifacts),
            token_expires=self.token_expires,
            today=self.today,
            stopped=stopped,
            workdir=self.workdir.name or 'plurai_sim',
        )

    def empty_run(self, *, with_ids: bool = False) -> tuple[Run, dict[str, Any]]:
        """A run that measured nothing, for a report that has only a stop to show."""
        trials = self.offline.config.trials if self.offline is not None else 1
        doc = {
            'schema': RUN_SCHEMA,
            **self.identity(with_ids=with_ids),
            'evals': [],
            'trials': {'schema': 'trials-v1', 'n': trials, 'measurement': [], 'scenarios': {}},
        }
        return read_run(doc), doc

    # --- outputs --------------------------------------------------------------------------------------------

    def write_diff_files(self, diff: SuiteDiff) -> None:
        if diff.moved:
            write_atomic(self.out_dir / 'suite-diff.md', diff.full)
            write_atomic(self.out_dir / 'suite-diff-comment.md', diff.short)

    def write_outputs(self, run: Run, options: ReportOptions, doc: dict[str, Any] | None) -> None:
        # The results bodies carry no suite diff: the action appends suite-diff.md to the summary and
        # suite-diff-comment.md to the comment. report.md is whole, so the diff sits after its results.
        write_atomic(self.out_dir / 'summary.md', render_summary(run, options))
        write_atomic(self.out_dir / 'comment.md', render_comment(run, options))
        write_atomic(self.out_dir / 'report.md', render_report(run, options, suite_diff=self.diff.full if self.diff is not None and self.diff.moved else None))
        if doc is not None:
            upload = run_json_for_upload(doc, include_content=self.include_content, visibility=self.visibility)
            write_atomic(self.out_dir / 'run.json', json.dumps(upload, indent=2, sort_keys=True) + '\n')
            write_atomic(self.out_dir / 'junit.xml', render_junit(run, include_content=self.content_included))
            if self.content_included:
                self.copy_transcripts()

    def copy_transcripts(self) -> None:
        for index, base in enumerate(self.state.pass_dirs, start=1):
            trials = self.offline.config.trials if self.offline is not None else 1
            for trial in range(1, trials + 1):
                for path in sorted(run_doc.trial_dir(base, trial).glob('*/transcript.json')):
                    target = self.out_dir / _TRANSCRIPTS / path.parent.name / f'pass{index}-trial{trial}.json'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)

    def end(self, stopped: Stopped, *, doc: dict[str, Any] | None = None, run: Run | None = None, message: str | None = None) -> _Finished:
        """End the run at a stop: write the report (and ``run.json`` for a run that measured something) and give the exit code.

        ``message`` replaces the one stderr line the stop prints, for a stop whose own text ends in a command to copy."""
        self.disarm()
        if run is None:
            run, _ = self.empty_run(with_ids=True)
        options = self.options(stopped=stopped, measured=doc is not None)
        self.write_outputs(run, options, doc)
        print(message or f'plurai ci run: {stopped.log_line}', file=sys.stderr)
        return _Finished(stopped.exit_code)

    def end_upgrade_required(self, refusal: client_handshake.UpgradeRefusal, *, doc: dict[str, Any] | None = None, run: Run | None = None) -> _Finished:
        """End the run on a version refusal: the handshake module's text to the log and the reports, and the typed refusal in
        ``run.json``. A refusal after results (``doc`` and ``run``) keeps them; one before any thread writes a run with no
        cells."""
        measured = doc is not None and run is not None
        if doc is None or run is None:
            run, doc = self.empty_run(with_ids=True)
        doc['refusal'] = refusal.record()
        text = client_handshake.upgrade_message(refusal)
        stopped = Stopped.upgrade_required(refusal, after_results=measured)
        return self.end(stopped, doc=doc, run=run, message=f'plurai ci run: {text}')

    def end_run(self, run: Run, doc: dict[str, Any]) -> int:
        self.disarm()
        options = self.options()
        self.write_outputs(run, options, doc)
        result = verdict(run, options)
        return result.exit_code

    # --- interruption ---------------------------------------------------------------------------------------

    def on_signal(self, signalled: Signalled) -> int:
        """End the run after a signal; runs on the supervisor thread while the run itself is frozen. Cancels every open thread and
        stops the agent children under one deadline counted from the signal, then writes the partial results and removes the
        scratch directory. Returns the exit code."""
        log = self.state.threads
        ends = signalled.at + WORK_DEADLINE_SECONDS
        cancels = None
        if self.endpoint is not None:
            cancels = Cancels(self.endpoint.url, self.endpoint.token or '', ends=ends)
            for thread_id in log.snapshot():
                cancels.add(thread_id)
        killed = stop_children(CHILD_GRACE_SECONDS, signalled.hurry, ends=ends)
        released = cancels.wait(signalled.hurry) if cancels is not None else Released(0, 0)
        if released.unconfirmed:
            self.notes.append(
                f'{_plural_threads(released.unconfirmed)} could not be confirmed cancelled; the backend releases a thread when it has been idle long enough'
            )
        open_threads = max(released.confirmed + released.unconfirmed, len(log.snapshot()))
        stopped = Stopped.cancelled()
        code = self.write_cancelled(stopped)
        if self.state.scratch is not None:
            shutil.rmtree(self.state.scratch, ignore_errors=True)
        print(
            f'plurai ci run: {released.confirmed} of {open_threads} open threads cancelled, {killed} agent children killed '
            f'(signal {signal.Signals(signalled.number).name}, {time.monotonic() - signalled.at:.1f} s)',
            file=sys.stderr,
        )
        return code

    def write_cancelled(self, stopped: Stopped) -> int:
        """Write the reports of a cancelled run: the partial results when something was judged, else the stop alone. A failure
        while composing the partial results falls back to the stop alone, so summary.md and comment.md exist whatever happens."""
        try:
            snapshot = self.state.snapshot() if self.state.snapshot is not None else None
            if snapshot is not None:
                run, doc = snapshot
                self.write_outputs(run, self.options(), doc)
                print(f'plurai ci run: {stopped.log_line}', file=sys.stderr)
                return stopped.exit_code
        except Exception:  # noqa: BLE001 - the stop alone is still a report
            traceback.print_exc()
        try:
            return self.end(stopped).code
        except OSError:
            traceback.print_exc()
            return stopped.exit_code

    # --- steps ----------------------------------------------------------------------------------------------

    def step_arguments(self) -> None:
        args = self.args
        flag = problem = None
        if not _REPO.fullmatch(args.repo):
            flag, problem = '--repo', f'`--repo` is owner/name, such as `acme/support-agent`; got {args.repo[:60]!r}'
        elif args.selection not in SELECTIONS:
            flag, problem = '--selection', f'`--selection` is `smoke` or `full`; got {args.selection[:40]!r}'
        elif args.visibility is not None and args.visibility not in VISIBILITIES:
            flag, problem = '--visibility', f'`--visibility` is `public`, `private` or `internal`; got {args.visibility[:40]!r}'
        elif args.include_content is not None and args.include_content not in ('true', 'false'):
            flag, problem = '--include-content', f'`--include-content` is `true` or `false`; got {args.include_content[:40]!r}'
        elif args.run_url is not None and not is_plain_web_address(args.run_url):
            flag, problem = '--run-url', f'`--run-url` is a plain http(s) address with no spaces or markdown; got {args.run_url[:80]!r}'
        if problem is not None:
            raise self.end(Stopped.refused(f'{flag} is not valid', problem))

    def step_offline(self) -> None:
        try:
            self.offline = check_offline(self.workdir)
        except Refusal as refusal:
            raise self.end(Stopped.refused(refusal.heading, str(refusal))) from refusal

    def step_secrets(self) -> None:
        assert self.offline is not None  # noqa: S101 - set by the step before
        if not self.env.get('PLURAI_TOKEN', '').strip():
            self.no_token_diff()
            raise self.end(Stopped.no_secrets('PLURAI_TOKEN is not set'))
        for name in self.offline.agent_env:
            if not self.env.get(name, '').strip():
                raise self.end(Stopped.empty_env(name))

    def step_endpoint(self) -> Endpoint:
        """The backend and token, through the SDK's one resolver: https unless loopback, no userinfo, a token of the right shape.
        Nothing is sent to a backend that fails it, so a mistyped or plain-http URL never carries the token."""
        pair = {endpoints.URL_ENV: self.env.get(endpoints.URL_ENV, ''), endpoints.TOKEN_ENV: self.env.get(endpoints.TOKEN_ENV, '')}
        try:
            return endpoints.resolve(pair)
        except endpoints.EndpointError as error:
            raise self.end(Stopped.refused('the backend address or token is not usable', str(error))) from error

    def no_token_diff(self) -> None:
        """With no token the content diff cannot be fetched: moved ids are shown by id alone."""
        assert self.offline is not None  # noqa: S101
        if not self.args.base:
            return
        try:
            base_ids = self.base_ids = base_ids_at(self.workdir, self.args.base, self.offline.config)
        except BaseUnreadable as error:
            self.notes.append(f'the suite diff could not be computed ({error})')
            return
        diff = render_ids_only(base_ids, ids_map(self.offline.ids), 'ids differ from base; the content diff needs a token', base_ref=self.args.base[:12])
        self.diff = diff
        self.write_diff_files(diff)
        if diff.moved:
            self.notes.append('ids differ from base; the content diff needs a token')

    def step_online(self, backend: Backend) -> None:
        assert self.offline is not None  # noqa: S101
        expiry = backend.key_expiry(self.args.repo)
        self.token_expires = expiry.expires
        if expiry.note:
            self.notes.append(expiry.note)
        self.suite = check_ids(backend, self.offline.ids)
        self.credit_state = check_credits(backend)
        if self.credit_state is None:
            self.notes.append('the credit state was not checked: the backend exposes none, so a refusal at the first thread stops the run')
        check_smoke_names(self.suite, self.offline.config.smoke)
        for drifted in self.suite.drifted():
            self.notes.append(f'the content of id {drifted} changed on the backend since it was published')
        self.names = suite_names(backend, ids_map(self.offline.ids))

    def step_suite_diff(self, backend: Backend) -> None:
        assert self.offline is not None  # noqa: S101
        if not self.args.base:
            return
        try:
            base_ids = self.base_ids = base_ids_at(self.workdir, self.args.base, self.offline.config)
        except BaseUnreadable as error:
            self.notes.append(f'the suite diff could not be computed ({error})')
            return
        head_ids = ids_map(self.offline.ids)
        if base_ids == head_ids:
            return
        diffed: Diffed = diff_against(backend, head_ids, base_ids, base_ref=self.args.base[:12], include_content=self.content_included)
        self.notes.extend(diffed.notes)
        self.diff = diffed.diff
        self.write_diff_files(diffed.diff)

    def step_main_run(self) -> None:
        if not self.args.main_run:
            return
        assert self.offline is not None  # noqa: S101
        try:
            doc = json.loads(Path(self.args.main_run).read_text(encoding='utf-8'))
            self.main = read_run(doc, fail_on=self.offline.config.fail_on)
        except FileNotFoundError:
            self.main, self.main_error = None, "main's run.json is missing"
        except OSError as failure:
            self.main, self.main_error = None, f"main's run.json cannot be read ({type(failure).__name__})"
        except RunJsonError as failure:
            self.main, self.main_error = None, f"main's {failure}"
        except ValueError:
            self.main, self.main_error = None, "main's run.json is not valid JSON"
        except (KeyError, TypeError):
            self.main, self.main_error = None, "main's run.json is not a run this version reads"

    # --- the run --------------------------------------------------------------------------------------------

    def select(self) -> tuple[list[str], list[str]]:
        """The selected scenarios and, of those, the ones the artifact read stamps unsolvable."""
        assert self.offline is not None  # noqa: S101
        assert self.suite is not None  # noqa: S101
        names = list(self.suite.scenario_names)
        if self.args.selection == 'full':
            chosen = names
        else:
            wanted = set(self.offline.config.smoke)
            chosen = [n for n in names if n in wanted or n.startswith(REGRESSION_PREFIX)]
        unsolvable = set(self.suite.scenarios.unsolvable)
        return chosen, [n for n in chosen if n in unsolvable]

    def gating_scenarios(self, scenarios: list[str], plan: OverridePlan) -> list[str]:
        """The scenarios with a gating scenario × policy that no rule ignores: the ones the check counts after the run, so the
        quarantine cap says the same before it."""
        assert self.offline is not None  # noqa: S101
        assert self.suite is not None  # noqa: S101
        fail_on = self.offline.config.fail_on
        return [
            n
            for n in scenarios
            if any(plan.rule_for(n, name) is None and level_of(tags_of(tier, n), fail_on) == 'gating' for name, tier in self.suite.policies)
        ]

    def run_passes(self, endpoint: Endpoint, scenarios: list[str], quarantined: list[str], plan: OverridePlan) -> int:
        assert self.offline is not None  # noqa: S101
        assert self.suite is not None  # noqa: S101
        config, suite = self.offline.config, self.suite
        identity = self.identity(with_ids=True)
        mock_url = f'{endpoint.url.rstrip("/")}/mock/{self.offline.ids.env}/'
        version = agent_version(self.workdir)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix='plurai-ci-') as scratch:
            self.state.scratch = Path(scratch)
            finished: list[dict[str, Any]] = []
            retrying: list[list[str]] = []

            def snapshot() -> tuple[Run, dict[str, Any]] | None:
                """The run so far: the first pass whole, with the retry pass merged in when one is under way, else the first pass as far
                as it got."""
                elapsed = round(time.monotonic() - started)
                opened = len(self.state.threads.opened())
                if finished:
                    partial: dict[str, Any] | None = {**finished[0], 'duration_seconds': elapsed}
                    if retrying:
                        second = run_doc.partial_doc(
                            Path(scratch) / 'pass2',
                            scenarios=retrying[0],
                            trials=config.trials,
                            identity=identity,
                            evals=suite.policies,
                            threads=len(retrying[0]) * config.trials,
                            duration_seconds=elapsed,
                        )
                        if second is not None:
                            with contextlib.suppress(ValueError):
                                partial = merge_retry(partial, second, fail_on=config.fail_on)
                else:
                    partial = run_doc.partial_doc(
                        Path(scratch) / 'pass1',
                        scenarios=scenarios,
                        trials=config.trials,
                        identity=identity,
                        evals=suite.policies,
                        pre_quarantined=quarantined,
                        threads=opened,
                        duration_seconds=elapsed,
                    )
                if partial is None:
                    return None
                partial['cancelled'] = True
                return apply_overrides(read_run(partial, fail_on=config.fail_on), plan), partial

            self.state.snapshot = snapshot
            first = self.one_pass(1, Path(scratch) / 'pass1', endpoint=endpoint, scenarios=scenarios, mock_url=mock_url, version=version)
            doc = run_doc.pass_doc(
                Path(scratch) / 'pass1',
                trials=config.trials,
                identity=identity,
                evals=suite.policies,
                pre_quarantined=quarantined,
                threads=len(scenarios) * config.trials,
                duration_seconds=round(time.monotonic() - started),
            )
            if doc is None:
                if first == exit_codes.UPGRADE_REQUIRED:
                    raise self.end_upgrade_required(_RUNNER_REFUSAL)
                raise self.end(
                    Stopped.not_measured(
                        'infra', run_doc.runner_stop_words(Path(scratch) / 'pass1', first), 'run the job again; the job log has the runner output.'
                    )
                )
            self.stop_on_refusals(Path(scratch) / 'pass1', doc)
            finished.append(doc)
            run = apply_overrides(read_run(doc, fail_on=config.fail_on), plan)
            selection = select_retries(run)
            if selection.over_cap:
                return self.end_without_retry(run, doc, retry_stop(selection), cap_reason(selection))
            if selection.scenarios:
                retrying.append(list(selection.scenarios))
                second_exit = self.one_pass(
                    2, Path(scratch) / 'pass2', endpoint=endpoint, scenarios=list(selection.scenarios), mock_url=mock_url, version=version
                )
                second = run_doc.pass_doc(
                    Path(scratch) / 'pass2',
                    trials=config.trials,
                    identity=identity,
                    evals=suite.policies,
                    threads=len(selection.scenarios) * config.trials,
                    duration_seconds=round(time.monotonic() - started),
                )
                if second is None:
                    if second_exit == exit_codes.UPGRADE_REQUIRED:
                        raise self.end_upgrade_required(_RUNNER_REFUSAL, doc=doc, run=run)
                    why = f'the retry stopped with exit {second_exit}'
                    return self.end_without_retry(run, doc, Stopped.not_measured('infra', why, 'run the job again.'), why)
                self.stop_on_refusals(Path(scratch) / 'pass2', second)
                doc = merge_retry(doc, second, fail_on=config.fail_on)
                doc['duration_seconds'] = round(time.monotonic() - started)
                run = apply_overrides(read_run(doc, fail_on=config.fail_on), plan)
            return self.end_run(run, doc)

    def end_without_retry(self, run: Run, doc: dict[str, Any], stop: Stopped, why: str) -> int:
        """End the run when the retry cannot happen. A gating failure of the first pass is the result: the run exits on it, shows
        the measured results and names the retry problem in a note. Only when nothing gating failed is the problem the whole
        story, a bare stop."""
        if tally(run).failed:
            self.notes.append(retry_note(why))
            return self.end_run(run, doc)
        raise self.end(stop, doc=doc, run=run)

    def one_pass(self, index: int, out: Path, *, endpoint: Endpoint, scenarios: list[str], mock_url: str, version: str) -> int:
        assert self.offline is not None  # noqa: S101
        out.mkdir(parents=True, exist_ok=True)
        self.state.pass_dirs.append(out)
        plan = PassPlan(
            self.offline.simulate,
            self.offline.ids.scenarios,
            self.offline.ids.evals,
            scenarios,
            self.offline.config.trials,
            out,
            mock_url,
            f'ci-{self.state.token}-pass{index}',
        )
        return run_pass(plan, endpoint, version=version)

    def stop_on_refusals(self, out: Path, doc: dict[str, Any]) -> None:
        """A 426, 401 or 403 from any thread route ends the run as a refused version, or a rejected or refused token; none
        is retried."""
        assert self.offline is not None  # noqa: S101
        for _trial, _scenario, error in run_doc.attempt_errors(out, self.offline.config.trials):
            refusal = run_doc.upgrade_refusal(error)
            status = run_doc.credentials_status(error)
            if refusal is None and status is None:
                continue
            run = read_run(doc, fail_on=self.offline.config.fail_on)
            if refusal is not None:
                raise self.end_upgrade_required(refusal, doc=doc, run=run)
            if status == 401:
                raise self.end(Stopped.token_rejected(self.args.repo), doc=doc, run=run)
            route = str(error.get('request_url') or 'a thread route').split('?')[0].split('://', 1)[-1].partition('/')[2]
            raise self.end(Stopped.token_refused('/' + route if route else 'a thread route'), doc=doc, run=run)


def run_ci(args: RunArgs, env: Mapping[str, str], workdir: Path, *, sleep: Callable[[float], None] = time.sleep, today: date | None = None) -> int:
    """Run ``plurai ci run``. Returns the exit code: 0, 1, 10, 20, 65 or 77."""
    raw_out = env.get('PLURAI_CI_OUT_DIR', '').strip()
    if not raw_out:
        print('plurai ci run: PLURAI_CI_OUT_DIR is not set; name the directory the reports are written into', file=sys.stderr)
        return exit_codes.REFUSED
    out_dir = Path(raw_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_stale_outputs(out_dir)
    run_origin.mark_ci()
    state = RunState()
    thread_log.install(state.threads)
    ctx = _Run(args, env, workdir.resolve(), out_dir, today or today_utc(), sleep, state)
    guard = SignalGuard(ctx.on_signal)
    ctx.disarm = guard.disarm
    try:
        with guard:
            try:
                return _execute(ctx, sleep=sleep)
            finally:
                guard.disarm()
    except Interrupted as interrupted:
        return interrupted.code
    finally:
        thread_log.install(None)


def _execute(ctx: _Run, *, sleep: Callable[[float], None]) -> int:
    """The steps of the run, in order. Every path that ends the run has written its reports."""
    args = ctx.args
    backend: Backend | None = None
    try:
        ctx.step_arguments()
        ctx.step_offline()
        ctx.step_secrets()
        assert ctx.offline is not None  # noqa: S101
        endpoint = ctx.endpoint = ctx.step_endpoint()
        backend = Backend(endpoint.url, endpoint.token or '', sleep=sleep)
        ctx.step_main_run()
        try:
            backend.resume_seconds = RESUME_SECONDS
            try:
                ctx.step_online(backend)
            finally:
                backend.resume_seconds = 0.0
            ctx.step_suite_diff(backend)
            wait_for_mock(backend, ctx.offline.ids.env, sleep=sleep)
            return _run_selected(ctx, endpoint)
        except StopRun as stop:
            raise ctx.end(stop.stopped) from stop
        except UpgradeRefused as error:
            raise ctx.end_upgrade_required(error.refusal) from error
        except (TokenRejected, TokenRefused, BackendUnreachable) as error:
            raise ctx.end(stop_for(error, repo=args.repo)) from error
    except _Finished as finished:
        return finished.code
    except Exception:  # noqa: BLE001 - this command exits 0, 1, 10, 20, 65 or 77 only: a defect is reported and the run is not measured
        traceback.print_exc()
        stopped = Stopped.not_measured(
            'infra', 'the SDK hit an unexpected error', 'run the job again; the job log has the traceback, which Plurai needs if it repeats.'
        )
        try:
            return ctx.end(stopped).code
        except OSError:
            return stopped.exit_code
    finally:
        if backend is not None:
            backend.close()


def _nothing_selected(chosen: list[str], smoke: tuple[str, ...]) -> Stopped:
    """The selection leaves no scenario to run, so nothing would be measured and a green check would be false."""
    if chosen:
        detail = 'every selected scenario is ignored by an override in ci.yaml'
        fix = 'narrow the `overrides` in ci.yaml so that at least one scenario × policy runs.'
    else:
        smoke_state = 'names no scenario that the committed scenario set has' if smoke else 'is empty'
        detail = f'ci.yaml `smoke` {smoke_state}, and the committed scenario set has no `{REGRESSION_PREFIX}` scenario'
        fix = f'add scenario names to `smoke` in ci.yaml, or add a scenario named `{REGRESSION_PREFIX}…` to the scenario set.'
    return Stopped.not_measured('suite', 'nothing selected', fix, detail=detail)


def _run_selected(ctx: _Run, endpoint: Endpoint) -> int:
    assert ctx.offline is not None  # noqa: S101
    assert ctx.suite is not None  # noqa: S101
    config = ctx.offline.config
    chosen, quarantined = ctx.select()
    runnable = [n for n in chosen if n not in quarantined]
    policies = [name for name, _ in ctx.suite.policies]
    plan = plan_overrides(config.rules, scenarios=runnable, policies=policies, today=ctx.today, smoke=config.smoke, suite_scenarios=ctx.suite.scenario_names)
    to_run = list(plan.run_scenarios)
    if stop := quarantine_stop(ctx.gating_scenarios(to_run, plan), quarantined):
        raise ctx.end(stop)
    if not to_run and not quarantined:
        raise ctx.end(_nothing_selected(chosen, config.smoke))
    if not to_run:
        doc = {
            'schema': RUN_SCHEMA,
            **ctx.identity(with_ids=True),
            'evals': [{'id': n, 'verdict_tier': t} for n, t in ctx.suite.policies],
            'trials': {
                'schema': 'trials-v1',
                'n': config.trials,
                'measurement': [],
                'scenarios': {
                    n: {'invalid': [{'trial': i, 'reason': 'quarantined'} for i in range(1, config.trials + 1)], 'policies': {}} for n in quarantined
                },
            },
            'threads': 0,
        }
        run = apply_overrides(read_run(doc, fail_on=config.fail_on), plan)
        return ctx.end_run(run, doc)
    return ctx.run_passes(endpoint, to_run, quarantined, plan)


__all__ = ['SELECTIONS', 'VISIBILITIES', 'RunArgs', 'RunState', 'run_ci', 'write_atomic']
