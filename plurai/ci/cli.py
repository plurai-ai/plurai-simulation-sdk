"""The command lines of ``plurai ci run``, ``plurai ci validate``, ``plurai ci token`` and ``plurai suite diff``.

``ci run`` takes its token from ``PLURAI_TOKEN`` and its backend from ``PLURAI_URL``, production when that is unset, and writes
into ``PLURAI_CI_OUT_DIR``; it reads no GitHub token and no GitHub event. ``PLURAI_ACTION_VERSION``, when the action sets it, is the
action version in ``run.json``.
Its exit codes are 0 (every counted scenario × policy held), 1 (one failed), 10 (the backend refuses this SDK version),
20 (not measured), 65 (refused) and 77 (token rejected or refused). ``ci validate`` exits 0, 10, 20, 65 or 77 and ``suite diff``
0, 10, 65 or 77; ``ci token`` adds 17, 69 and 75 (:mod:`plurai.ci.token`).

Two exits differ from the table in :mod:`plurai.exit_codes`. A bad command line of ``ci`` or ``suite`` exits 65 where every other
command exits 64, so a CI step sees one set of codes; and a Ctrl-C of ``ci token`` exits 1 where a run is cancelled with 20. A
Ctrl-C of ``ci run`` is the cancel: it exits 20 with the partial reports, never 130.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from plurai import _endpoints, exit_codes
from plurai.ci import suite as suite_module
from plurai.ci import token as token_module
from plurai import client_handshake
from plurai.ci.backend import Backend, BackendUnreachable, TokenRefused, TokenRejected, UpgradeRefused
from plurai.ci.gitrepo import merge_base_with_default_branch
from plurai.ci.config import Refusal, check_offline, today_utc
from plurai.ci.online import StopRun, check_credits, check_ids, check_smoke_names, stop_for
from plurai.ci.overrides import plan_overrides
from plurai.ci.run import SELECTIONS, VISIBILITIES, RunArgs, run_ci
from plurai.report import Stopped, override_note_text


def _stop(stopped: Stopped) -> int:
    print(f'plurai: {stopped.reason}. {stopped.fix}', file=sys.stderr)
    return stopped.exit_code


def _upgrade_stop(error: UpgradeRefused) -> int:
    print(f'plurai: {client_handshake.upgrade_message(error.refusal)}', file=sys.stderr)
    return exit_codes.UPGRADE_REQUIRED


def _cmd_run(args: argparse.Namespace) -> int:
    return run_ci(
        RunArgs(args.repo, args.selection, args.base, args.main_run, args.visibility, args.include_content, args.run_url),
        os.environ,
        Path.cwd(),
    )


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        offline = check_offline(Path.cwd())
    except Refusal as refusal:
        print(f'plurai ci validate: refused: {refusal}', file=sys.stderr)
        return exit_codes.REFUSED
    if not args.online:
        print('ci.yaml and the committed ids are valid (offline rules).')
        return exit_codes.OK
    pair = {name: os.environ.get(name, '') for name in (_endpoints.URL_ENV, _endpoints.TOKEN_ENV)}
    if not pair[_endpoints.TOKEN_ENV].strip():
        return _stop(Stopped.not_measured('infra', 'no token for the online rules', 'set PLURAI_TOKEN and run it again.'))
    try:
        endpoint = _endpoints.resolve(pair)
    except _endpoints.EndpointError as error:
        print(f'plurai ci validate: refused: {error}', file=sys.stderr)
        return exit_codes.REFUSED
    backend = Backend(endpoint.url, endpoint.token or '')
    try:
        suite = check_ids(backend, offline.ids)
        check_credits(backend)
        check_smoke_names(suite, offline.config.smoke)
    except StopRun as stop:
        return _stop(stop.stopped)
    except UpgradeRefused as error:
        return _upgrade_stop(error)
    except (TokenRejected, TokenRefused, BackendUnreachable) as error:
        return _stop(stop_for(error))
    finally:
        backend.close()
    plan = plan_overrides(
        offline.config.rules,
        scenarios=suite.scenario_names,
        policies=[name for name, _ in suite.policies],
        today=today_utc(),
        smoke=offline.config.smoke,
    )
    for note in plan.notes:
        print(f'note: {override_note_text(note)}')
    print('ci.yaml, the committed ids and the backend agree (online rules).')
    return exit_codes.OK


def _cmd_token(args: argparse.Namespace) -> int:
    return token_module.run(args.github, args.expires_in, os.environ)


def _cmd_suite_diff(args: argparse.Namespace) -> int:
    workdir = Path.cwd()
    try:
        offline = check_offline(workdir)
    except Refusal as refusal:
        print(f'plurai suite diff: refused: {refusal}', file=sys.stderr)
        return exit_codes.REFUSED
    base = args.base or merge_base_with_default_branch(workdir)
    if not base:
        print('plurai suite diff: no base: pass --base <ref>, or run it in a repository with a default branch', file=sys.stderr)
        return exit_codes.REFUSED
    try:
        base_ids = suite_module.base_ids_at(workdir, base, offline.config)
    except suite_module.BaseUnreadable as error:
        print(f'plurai suite diff: refused: {error}', file=sys.stderr)
        return exit_codes.REFUSED
    head_ids = suite_module.ids_map(offline.ids)
    if base_ids == head_ids:
        print('The committed ids are the same as at the base: no suite changes.')
        return exit_codes.OK
    try:
        endpoint = _endpoints.resolve()
    except _endpoints.EndpointError as error:
        print(f'plurai suite diff: {error}', file=sys.stderr)
        return exit_codes.CREDENTIALS
    backend = Backend(endpoint.url, endpoint.token or '')
    try:
        diffed = suite_module.diff_against(backend, head_ids, base_ids, base_ref=base[:12], include_content=True)
    except UpgradeRefused as error:
        return _upgrade_stop(error)
    except (TokenRejected, TokenRefused) as error:
        return _stop(stop_for(error))
    finally:
        backend.close()
    print(diffed.diff.full, end='')
    for note in diffed.notes:
        print(f'note: {note}', file=sys.stderr)
    return exit_codes.OK


def _days(text: str) -> int:
    try:
        return token_module.expires_in_days(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


#: The commands whose bad command line is a refusal (65), not a usage error (64): a CI step reads one set of codes.
COMMANDS: tuple[list[str], ...] = (['ci'], ['suite'])


def add_parsers(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register ``ci run``, ``ci validate``, ``ci token`` and ``suite diff`` on the ``plurai`` parser."""
    ci = sub.add_parser('ci', help='Run the simulation suite in CI, or validate its configuration.')
    verbs = ci.add_subparsers(dest='verb', required=True)

    run = verbs.add_parser(
        'run',
        help='Run the committed suite and write the reports into $PLURAI_CI_OUT_DIR.',
        description='Run from the simulation directory (plurai_sim/). The token is PLURAI_TOKEN, the backend production unless PLURAI_URL names another, '
        'and the output directory PLURAI_CI_OUT_DIR. Files written there: summary.md, comment.md, report.md, run.json and junit.xml (a run that '
        'measured something), suite-diff.md and suite-diff-comment.md (when committed ids moved) and transcripts/ (when content '
        'is included). PLURAI_ACTION_VERSION, set by the simulation action, is recorded as the action version in run.json and the report '
        'footer. Reads no GitHub token and no GitHub event. Exit codes: 0 every counted scenario x policy held; 1 one failed; '
        '10 the backend refuses this SDK version; 20 not measured (infra, agent, suite, credits or no secrets); 65 refused, a bad command line included; '
        '77 token rejected or refused. SIGINT and SIGTERM cancel the open threads, '
        'stop the agent, write the partial reports and exit 20.',
    )
    run.add_argument(
        '--repo', required=True, metavar='OWNER/REPO', help='The repository: names the run and the service-account key whose expiry the report shows.'
    )
    run.add_argument(
        '--selection',
        required=True,
        metavar='|'.join(SELECTIONS),
        help='smoke runs the ci.yaml smoke list plus every regression_* scenario; full runs the whole committed scenario set.',
    )
    run.add_argument('--base', metavar='SHA', default=None, help='The base commit for the suite diff. Without it no suite diff is made.')
    run.add_argument(
        '--main-run', metavar='PATH', default=None, help="Main's latest completed run.json, for the vs-main line. Without a usable file the report says so."
    )
    run.add_argument(
        '--visibility',
        metavar='|'.join(VISIBILITIES),
        default=None,
        help='Repository visibility: with no --include-content, run content (judge text, transcripts) is included for private only.',
    )
    run.add_argument(
        '--include-content', metavar='true|false', default=None, help='Include run content in the reports and uploads. Absent: decided from --visibility.'
    )
    run.add_argument(
        '--run-url',
        metavar='URL',
        default=None,
        help="The run's own page (on GitHub Actions: https://github.com/OWNER/REPO/actions/runs/ID/attempts/N). The PR comment links its "
        'job summary phrases there. Without it they stay plain text.',
    )
    run.set_defaults(func=_cmd_run)

    validate = verbs.add_parser(
        'validate',
        help="Check ci.yaml and the committed ids (offline), and with --online the backend's view of them.",
        description='Offline: the five ci.yaml keys, tags, override selectors and ties, the smoke list, agent.env names, id files, the env '
        'alias and the SDK pin; exit 65 with a message that names the fix. --online also needs PLURAI_TOKEN (and PLURAI_URL for another backend than '
        'production): ids exist and link within their lineage, credit, smoke names. Exit codes: 0 valid; 65 refused, a bad command line included; '
        '20 no token or an unreachable backend; 77 token rejected or refused; 10 the backend refuses this SDK version.',
    )
    validate.add_argument('--online', action='store_true', help='Also run the rules that need the backend.')
    validate.set_defaults(func=_cmd_validate)

    token = verbs.add_parser(
        'token',
        help='Mint a service-account key for a repository and write it to its PLURAI_TOKEN secret.',
        description='Run once per repository, after `plurai login` and `gh auth login`. Mints a service-account key with the user key '
        '`plurai login` saved, named OWNER/REPO, and writes it to the repository secret PLURAI_TOKEN through `gh secret set` on stdin. The key '
        'is never printed. The output names the key, its expiry date and the repository. The key has full access to your Plurai '
        'organization, because Plurai keys carry no narrower permission. Refuses (exit 65, nothing minted) without a saved login, with '
        'PLURAI_TOKEN set in the environment, or when `gh` is missing or not logged in. When `gh secret set` fails, the new key is revoked '
        'again and the exit status is 65; if the revoke fails too, the output names the key to revoke in the Plurai dashboard. '
        'Other exit codes: 77 the saved login is not accepted; 10 the backend refuses this SDK version; 75 try again later; '
        '69 the backend could not be reached; 17 the backend refused the request. A key named OWNER/REPO that already exists is refused with 65. '
        'Ctrl-C after the key may exist revokes it and exits 1, not 130.',
    )
    token.add_argument('--github', required=True, metavar='OWNER/REPO', help='The repository whose secret PLURAI_TOKEN receives the key.')
    token.add_argument(
        '--expires-in',
        metavar='DAYS',
        type=_days,
        default=token_module.DEFAULT_EXPIRES_IN_DAYS,
        help=f'Days until the key expires, {token_module.MIN_EXPIRES_IN_DAYS} to {token_module.MAX_EXPIRES_IN_DAYS} '
        f'(default {token_module.DEFAULT_EXPIRES_IN_DAYS}).',
    )
    token.set_defaults(func=_cmd_token)

    suite = sub.add_parser('suite', help="The suite's committed ids and what changed in them.")
    suite_verbs = suite.add_subparsers(dest='verb', required=True)
    diff = suite_verbs.add_parser(
        'diff',
        help='Print what changed in the suite since a git ref, rendered as CI renders it.',
        description='Compares the committed ids of the working tree with those at --base (default: the merge-base with the default branch) '
        "and prints the backend's content difference, run content included. Uses the backend of PLURAI_URL / PLURAI_TOKEN or the saved login.",
    )
    diff.add_argument('--base', metavar='REF', default=None, help='The git ref to compare with.')
    diff.set_defaults(func=_cmd_suite_diff)
