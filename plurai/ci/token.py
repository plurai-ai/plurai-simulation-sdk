"""``plurai ci token --github <owner>/<repo>``: mint a service-account key and write it to the repo secret ``PLURAI_TOKEN``.

The user key ``plurai login`` saved authorises ``POST /api-keys`` with ``identity: {type: "serviceAccount"}``. With no service
account id given, the backend creates the service account and a key bound to the org. The key is named ``<owner>/<repo>``, so
the key list and the expiry line of a CI report name the repo, and it expires 365 days out unless ``--expires-in`` says
otherwise. ``gh secret set PLURAI_TOKEN --repo <owner>/<repo>`` receives the key on stdin. The key is never printed, logged or
placed on a command line; the output names the key, its expiry date and the repo.

The user key is sent only to the key routes of the SDK's environment table (:mod:`plurai.login._environments`) for the host the
saved login names, never to the url the credentials file holds.

Refusals (exit 65, before any key exists): no usable saved login, ``PLURAI_TOKEN`` in the environment (that is a CI key, not a
user), ``PLURAI_URL`` naming another backend than the saved login, ``gh`` missing or not logged in, a repo that is not
``owner/name``, an expiry outside 2 to 3650 days. When ``gh secret set`` fails, the new key is revoked with the user key and
the command exits 65; when the revoke fails too, the output names the key id to revoke by hand. A backend answer maps to the
exit code of its status: 77 for 401 and 403, 10 for 426, 75 for 429, 69 for an unreachable or failing backend, 17 for another
4xx. A name that is taken exits 65, like the other refusals the user fixes.

Ctrl-C exits 1, not the shell's 130 (see :mod:`plurai.exit_codes`). While the secret is being written the key is revoked first;
while the key is being created no id is known, so the message says a key may exist and where to look.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlsplit

import httpx

from plurai import _http, client_handshake, credentials, exit_codes
from plurai._endpoints import TOKEN_ENV, URL_ENV, diagnostic_base
from plurai.ci.isotime import parse_instant
from plurai.login import _environments as environments
from plurai.login.cli import HTTP_TIMEOUT, _keys_url, _may_exist, _revoke_key

#: The repository secret the action reads the key from.
SECRET_NAME = 'PLURAI_TOKEN'  # noqa: S105 - the name of a secret, not a value

DEFAULT_EXPIRES_IN_DAYS = 365
#: The backend refuses an expiry less than a day out, so one day would race its own clock.
MIN_EXPIRES_IN_DAYS = 2
MAX_EXPIRES_IN_DAYS = 3650

#: The exit status of a Ctrl-C once a key may exist. It is not the shell's 130: the command checks no policy, and the CI commands
#: keep to their own codes (see :mod:`plurai.exit_codes`).
_CANCELLED = exit_codes.POLICIES_FAILED
_GH_TIMEOUT = 120.0
_REPO = re.compile(r'[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_.][A-Za-z0-9._-]{0,99}')
_GH_ERROR_LIMIT = 400


class Refused(Exception):  # noqa: N818 - a control-flow signal inside this module, not an error type of its API
    """The command cannot go on; the message is printed and ``code`` is the exit status."""

    def __init__(self, message: str, code: int = exit_codes.REFUSED) -> None:
        super().__init__(message)
        self.code = code


def expires_in_days(text: str) -> int:
    """The ``--expires-in`` value ``text`` as days; raises :class:`ValueError` outside the range the backend accepts."""
    try:
        days = int(text)
    except ValueError:
        raise ValueError(f'--expires-in takes a whole number of days, not {text!r}') from None
    if not MIN_EXPIRES_IN_DAYS <= days <= MAX_EXPIRES_IN_DAYS:
        raise ValueError(f'--expires-in takes {MIN_EXPIRES_IN_DAYS} to {MAX_EXPIRES_IN_DAYS} days, not {days}')
    return days


@dataclass(frozen=True)
class _Minted:
    """A key the backend created: its id, its secret (never printed) and the expiry the backend recorded."""

    key_id: str
    secret: str
    expires_at: datetime
    #: Whether ``expires_at`` is the date the backend recorded; ``False`` when it is the one this command asked for.
    expiry_confirmed: bool = True


def _say(text: str) -> None:
    print(f'plurai ci token: {text}', file=sys.stderr)


def _is_set(env: Mapping[str, str], name: str) -> bool:
    return bool((env.get(name) or '').strip())


def _check_repo(repo: str) -> str:
    if not _REPO.fullmatch(repo) or repo.endswith(('/.', '/..')):
        raise Refused(f'--github takes <owner>/<repo> (letters, digits, "-", "_" and "." only), not {credentials.printable(repo)!r}.')
    return repo


def _saved_login(env: Mapping[str, str]) -> tuple[credentials.Credential, environments.Environment]:
    """The usable saved login and the environment row that holds its key routes; refuses otherwise."""
    if _is_set(env, TOKEN_ENV):
        raise Refused(
            f'{TOKEN_ENV} is set in the environment. That is a CI key, not a person: the command mints a key with the user key that '
            f'`plurai login` saves. Unset {TOKEN_ENV} and run it again.'
        )
    if moved := credentials.home_problem(env):
        raise Refused(f'{moved}. Nothing was minted.')
    saved = credentials.read()
    if saved.problem:
        raise Refused(f'the saved login cannot be used ({credentials.printable(saved.problem)}). Run `plurai login`, then this command again.')
    if not saved.token:
        raise Refused('there is no saved login to mint the key with. Run `plurai login`, then this command again.')
    account = saved.account
    if account is not None and account.expires_at is not None:
        expiry = parse_instant(account.expires_at)
        if expiry is None:
            raise Refused('the saved login has an expiry date that cannot be read. Run `plurai login` to create a new one.')
        if expiry <= datetime.now(UTC):
            raise Refused('the saved login has expired. Run `plurai login` to create a new one, then this command again.')
    if _is_set(env, URL_ENV) and diagnostic_base((env.get(URL_ENV) or '').strip()) != diagnostic_base(saved.url):
        raise Refused(
            f"{URL_ENV} names another backend than the saved login. The key would belong to the saved login's backend; "
            f'unset {URL_ENV}, or run `plurai login` against the backend you mean.'
        )
    row = environments.for_host(urlsplit(saved.url).hostname or '')
    if row is None or row.unassigned():
        raise Refused(f'this SDK has no key routes for {credentials.printable(saved.url)}, so it cannot mint a key there.')
    return saved, row


def _find_gh(env: Mapping[str, str]) -> str:
    """The ``gh`` executable on ``PATH`` when it is logged in; refuses otherwise."""
    gh = shutil.which('gh', path=env.get('PATH'))
    if gh is None:
        raise Refused(
            'the GitHub CLI `gh` is not installed or not on PATH. Install it from https://cli.github.com, run `gh auth login`, then this command again.'
        )
    try:
        status = subprocess.run([gh, 'auth', 'status'], capture_output=True, text=True, env=dict(env), timeout=_GH_TIMEOUT, check=False)  # noqa: S603
    except (OSError, subprocess.TimeoutExpired) as failure:
        raise Refused(f'`gh auth status` could not run ({type(failure).__name__}). Run `gh auth login`, then this command again.') from None
    if status.returncode != 0:
        raise Refused('`gh` is not logged in. Run `gh auth login`, then this command again.')
    return gh


def _refuse_status(response: Any, repo: str) -> None:  # noqa: ANN401 - the httpx response
    """Raise for every answer to the creation request that is not a 2xx."""
    status = response.status_code
    if status == client_handshake.UPGRADE_STATUS:
        raise Refused(client_handshake.upgrade_message_for(response), exit_codes.UPGRADE_REQUIRED)
    if status in (401, 403):
        raise Refused(
            f'the backend did not accept the saved login (HTTP {status}). Run `plurai logout` and `plurai login`. A key made by this command '
            'cannot mint another one.',
            exit_codes.CREDENTIALS,
        )
    if status == 409:
        raise Refused(
            f'a key named {repo} already exists (HTTP 409); there is one key per repo. To replace it, revoke that key in the Plurai '
            'dashboard and run the command again.',
            exit_codes.REFUSED,
        )
    if status == 429:
        raise Refused('the backend asked to wait before creating another key (HTTP 429); run the command again in a moment.', exit_codes.TEMPFAIL)
    if not 200 <= status < 300:
        code = exit_codes.TRANSPORT if status >= 500 else exit_codes.BACKEND_REFUSED
        raise Refused(f'the backend did not create the key (HTTP {status}). Nothing was written.', code)


def _requested_expiry(days: int) -> datetime:
    return (datetime.now(UTC) + timedelta(days=days)).replace(microsecond=0)


def _expiry_from(body: object) -> datetime | None:
    """The expiry the backend recorded, or ``None`` when its answer carries none or one that cannot be read."""
    return parse_instant(body.get('expiresAt')) if isinstance(body, dict) else None


def _create_key(row: environments.Environment, bearer: str, repo: str, days: int) -> _Minted:
    """Create the service-account key; raises :class:`Refused` for every answer that did not give a usable key."""
    requested = _requested_expiry(days)
    payload = {'name': repo, 'identity': {'type': 'serviceAccount'}, 'expiresAt': requested.isoformat().replace('+00:00', 'Z')}
    try:
        with _http.Client(timeout=HTTP_TIMEOUT, follow_redirects=False) as client:
            response = client.post(_keys_url(row), json=payload, headers={'Authorization': f'Bearer {bearer}'})
    except KeyboardInterrupt:
        # The request may have reached the backend: no answer is not "no key".
        raise Refused(f'cancelled while the key was being created. {_may_exist(repo)}', _CANCELLED) from None
    except (httpx.ConnectError, httpx.ConnectTimeout) as failure:
        raise Refused(f'the backend could not be reached to create the key ({type(failure).__name__}). Nothing was written.', exit_codes.TRANSPORT) from None
    except httpx.HTTPError as failure:
        raise Refused(
            f'the connection to the backend failed while the key was being created ({type(failure).__name__}). {_may_exist(repo)}', exit_codes.TRANSPORT
        ) from None
    _refuse_status(response, repo)
    try:
        body = response.json()
    except ValueError:
        body = None
    secret, key_id = (body.get('secret'), body.get('id')) if isinstance(body, dict) else (None, None)
    known = key_id if isinstance(key_id, str) and key_id else None
    if known and not (isinstance(secret, str) and credentials.TOKEN_PATTERN.fullmatch(secret)):
        # A key exists whose secret cannot be stored as is: it is useless and has full org access, so it is revoked again. The
        # user key authorises the revoke, so the shape of the secret does not matter.
        if _revoke_key(row, known, bearer).outcome == 'revoked':
            raise Refused(
                'the backend created a key whose secret this SDK cannot use, and it has been revoked again. Nothing was written.', exit_codes.TRANSPORT
            )
    if not isinstance(secret, str) or not credentials.TOKEN_PATTERN.fullmatch(secret) or known is None:
        raise Refused(f'the backend answered without a usable key. {_may_exist(repo, known)}', exit_codes.TRANSPORT)
    recorded = _expiry_from(body)
    return _Minted(known, secret, recorded or requested, expiry_confirmed=recorded is not None)


def _scrub(text: str, secret: str) -> str:
    """``text`` as one short printable line without the key value."""
    shown = credentials.printable(' '.join(text.replace(secret, '***').split()))
    return shown if len(shown) <= _GH_ERROR_LIMIT else shown[:_GH_ERROR_LIMIT] + '...'


def _write_secret(gh: str, repo: str, secret: str, env: Mapping[str, str]) -> str | None:
    """Write ``secret`` to the repo secret through ``gh`` on stdin. ``None`` when it worked, else why it did not (never the key)."""
    command = [gh, 'secret', 'set', SECRET_NAME, '--repo', repo]
    try:
        done = subprocess.run(command, input=secret, capture_output=True, text=True, env=dict(env), timeout=_GH_TIMEOUT, check=False)  # noqa: S603
    except subprocess.TimeoutExpired:
        return f'`gh secret set` did not finish within {int(_GH_TIMEOUT)} seconds'
    except OSError as failure:
        return f'`gh secret set` could not run ({type(failure).__name__})'
    if done.returncode == 0:
        return None
    detail = _scrub(done.stderr or done.stdout, secret)
    return f'`gh secret set` failed (exit {done.returncode})' + (f': {detail}' if detail else '')


def _undo(row: environments.Environment, minted: _Minted, bearer: str, repo: str, why: str) -> str:
    """Revoke the key just made with the user key; the sentence says whether it worked and, if not, which key to revoke by hand."""
    try:
        revocation = _revoke_key(row, minted.key_id, bearer)
    except KeyboardInterrupt:
        return f'{why}. Cancelled while the key was being revoked: revoke the key {credentials.printable(minted.key_id)} ({repo}) in the Plurai dashboard.'
    if revocation.outcome == 'revoked':
        return f'{why}. The key just made ({credentials.printable(minted.key_id)}) was revoked again; no key was left.'
    return (
        f'{why}. The key just made could not be revoked ({revocation.problem}) and has full org access: revoke the key '
        f'{credentials.printable(minted.key_id)} ({repo}) in the Plurai dashboard.'
    )


def run(repo: str, days: int, env: Mapping[str, str]) -> int:
    """Mint the key for ``repo`` and write it to its secret; the exit status of the command."""
    try:
        repo = _check_repo(repo)
        saved, row = _saved_login(env)
        bearer = saved.token
        gh = _find_gh(env)
        minted = _create_key(row, bearer, repo, days)
        try:
            problem = _write_secret(gh, repo, minted.secret, env)
        except KeyboardInterrupt:
            raise Refused(_undo(row, minted, bearer, repo, 'cancelled while the secret was being written'), _CANCELLED) from None
        if problem is not None:
            raise Refused(_undo(row, minted, bearer, repo, f'{problem}; the secret was not written'))
    except Refused as refused:
        _say(str(refused))
        return refused.code
    print(f'Key: {repo} ({credentials.printable(minted.key_id)})')
    expires = minted.expires_at.astimezone(UTC).date().isoformat()
    print(
        f'Expires: {expires}'
        if minted.expiry_confirmed
        else f'Expires: {expires} (the date this command requested, not confirmed by the backend: check the key in the Plurai dashboard)'
    )
    print(f'secret written to {repo} ({SECRET_NAME}). The key value is not shown.')
    print(
        'The key has full access to your Plurai organization: anything the organization can do, not only runs for this repo. '
        'To rotate it, revoke the key in the Plurai dashboard and run this command again.'
    )
    return exit_codes.OK
