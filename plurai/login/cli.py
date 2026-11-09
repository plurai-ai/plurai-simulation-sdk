"""``plurai login`` and ``plurai logout``: sign in through the browser and save a backend key where every Plurai client reads it, and end that login.

    plurai login --status
    plurai logout

The backend is ``PLURAI_URL`` when it is set, else production; its host selects the row of the environment table
(:mod:`plurai.login._environments`), and any other host is refused. The command opens the browser on the backend's
sign-in and consent screen (the url is printed too, for a machine where no browser opens), creates a key at the backend
with the access token the sign-in returned, writes the key to the credentials file, revokes the sign-in's refresh token
and forgets both OAuth tokens; the refresh token is revoked when the key could not be created or saved too. The browser
tab is answered only after the file is written and read back: with a 303 to the environment's connected page. Nothing
secret is printed: not the key, not a token, not the authorization code.

A login that overwrites a saved key revokes that key, best effort and with the key itself, once the new key is saved and
read back; it needs the saved file to name the key (``account``) and the key's backend to be in the environment table. When
the revocation cannot run or fails, the login says so, names the old key and says to revoke it in the dashboard. A saved
file that cannot be used (open permissions, malformed) is overwritten with a warning that its key was not revoked; a link in
the file's place is refused, never followed.
Two logins at once: after writing, the file is read back, and a login whose key lost revokes its own key and says so. Every
login that wrote, whether its key won or lost, retires the key the file held just before its own write. A login that saves
after another therefore retires that login's key, and the loser still retires the key it replaced. One window is left
without a lock: two logins read the same key, and the first writes and reads back before the second writes. Both retire
that key, and the first login's key stays live with no file naming it. The window is the second login's read to write,
not the browser wait.

No login runs when none is needed: with ``PLURAI_TOKEN`` set the environment pair is what every client uses, so the
command refuses and says to unset it; with a usable saved key for the same backend it says so and exits 0.

A key the command created but could not save is revoked again with itself; when that fails the message names the key for the
dashboard. When the answer to the creation request is lost or unusable a key may exist, and the message says so; a key whose
secret the file cannot hold is revoked again with that secret. Ctrl-C ends the login with one line and exit 77: a key made
before the interrupt is revoked again unless the file already holds it. Once the file holds the new key, the key it
overwrote is still revoked; a Ctrl-C during that revocation names the key for the dashboard.

A login for a backend only ``PLURAI_URL`` names ends with a line saying that every other command refuses that
configuration until ``PLURAI_URL`` is unset or ``PLURAI_TOKEN`` is exported too.

On Linux without ``DISPLAY``, ``WAYLAND_DISPLAY`` or ``BROWSER`` the command does not start a browser: ``webbrowser`` would
fall back to a text browser that takes the terminal. It prints the url and waits. A set ``BROWSER`` names the opener the
user chose (remote editors set it to a helper that opens the url on the user's own machine), and ``webbrowser`` uses it.

``--status`` names the backend and the account of the saved key and never prints the token. A key that has expired, or
whose expiry date cannot be read, is no usable credential.

``logout`` revokes the saved key at the backend with the key itself, then removes the credentials file. It says "Revoked"
only when the backend confirms the revocation. When the backend answers 401 or 404 it no longer accepts the key, and the
command says that, not that it revoked it. When the revocation fails in a way a retry can fix (the backend is unreachable,
429, 5xx, an outdated client) the file is kept and the command can be run again. When no retry can help (the file does not
say which key it holds, the SDK has no key routes for its backend, 403) the file is removed, and the command names the key
that still works and says to revoke it in the dashboard.

Exit 0 ok; 77 (CREDENTIALS) consent denied, the sign-in cancelled with Ctrl-C (a person's decision, like a denial, so a
script does not retry it), the backend rejected the sign-in or it has no organization, or ``--status`` found no usable
credential, or ``logout`` removed the file without revoking a key that still works; 75 (TEMPFAIL) the sign-in
timed out, the authorization server was temporarily unavailable, a server answered 429 or a name conflict survived its retry,
or ``logout`` kept the file because the key could not be revoked yet; 10 (UPGRADE_REQUIRED) the backend refuses this SDK
version; 69 (TRANSPORT) a server was unreachable or answered something unusable, or the authorization server
failed; 65 (REFUSED) the backend, the home directory or the credentials file was refused; 64 (USAGE) a bad command line or
``PLURAI_TOKEN`` set; 17 (BACKEND_REFUSED) the backend answered the creation request with another 4xx, such as 400.
See :mod:`plurai.exit_codes`.
"""

from __future__ import annotations

import argparse
import os
import re
import secrets
import socket
import sys
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, NoReturn
from urllib.parse import quote, urlsplit

from plurai import client_handshake, credentials, exit_codes
from plurai._endpoints import TOKEN_ENV, URL_ENV, diagnostic_base
from plurai.login import _environments as environments

if TYPE_CHECKING:
    from plurai.login import oauth_client

_DASHBOARD = 'Create or revoke API keys in the Plurai dashboard; revoking the key there ends this login.'
_REVOKE_IN_DASHBOARD = 'Revoke it in the Plurai dashboard.'
_EXPOSURE = 'The file is readable by anything that runs as you, a shell a model runs included.'
_ORGANIZATION_HINT = (
    'Finish setting up your organization in the Plurai dashboard, then run `plurai login` again and choose that organization on the consent screen.'
)
#: A 401 on key creation with the sign-in as the bearer has two likely causes the answer does not tell apart.
_NOT_ACCEPTED_HINT = (
    'If your organization is not set up, finish setting up your organization in the Plurai dashboard, then run `plurai login` again and '
    'choose it on the consent screen. Otherwise this client may not be registered at the backend, or the sign-in may not have been granted '
    'the scope to create API keys; a newer release of the SDK, or the Plurai team, can fix that.'
)

#: What the sign-in asks for. ``api_keys.create`` is the consent to create the key; ``user:org:read`` makes the consent
#: screen offer the organization choice that reaches the access token as ``org_id``. The list is the backend's decision.
SCOPES: Final = ('openid', 'profile', 'email', 'api_keys.create', 'user:org:read')

#: The limit of each request to the backend, in seconds.
HTTP_TIMEOUT: Final = 15.0

#: Opens the authorize url; a test replaces it with a function that follows the url the way a browser does.
open_browser: Callable[[str], object] = webbrowser.open

#: ``sys.platform``, read through a name a test can replace.
_PLATFORM: str = sys.platform


def _can_open_browser(env: Mapping[str, str], platform: str | None = None) -> bool:
    """False on Linux without a display server and without ``BROWSER``: ``webbrowser`` then falls back to a text browser that takes the terminal."""
    name = _PLATFORM if platform is None else platform
    return not name.startswith('linux') or any(_is_set(env, variable) for variable in ('DISPLAY', 'WAYLAND_DISPLAY', 'BROWSER'))


def _no_browser(url: str) -> None:
    """The opener of a machine with no screen. It ignores the url on purpose: the printed url is the way in."""


class _Parser(argparse.ArgumentParser):
    """An argument error that does not repeat the arguments: a token typed there is already too exposed."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        print(f'{self.prog}: unrecognised or missing arguments. It takes no url and no token.', file=sys.stderr)
        sys.exit(exit_codes.USAGE)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--status', action='store_true', help='say which backend and account the saved key is for; never prints the key')


class _Failed(Exception):  # noqa: N818 - a control-flow signal inside this module, not an error type of its API
    """A login that cannot go on; ``code`` is its exit status and the message is printed."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


class _FailedAfterWrite(_Failed):
    """A login that fails after its write replaced ``previous``: that key is no longer in the file, so it is still retired."""

    def __init__(self, message: str, code: int, previous: credentials.Credential) -> None:
        super().__init__(message, code)
        self.previous = previous


class _CancelledAfterSave(_FailedAfterWrite):
    """A Ctrl-C after the file holds the new key: the key stays."""

    def __init__(self, message: str, previous: credentials.Credential) -> None:
        super().__init__(message, exit_codes.CREDENTIALS, previous)


def _refuse(text: str, code: int = exit_codes.REFUSED) -> int:
    print(f'plurai login: {text}', file=sys.stderr)
    return code


def _account_lines(saved: credentials.Credential) -> list[str]:
    """The lines that name the saved key's account. Every account string comes from the backend, so none is printed raw."""
    lines = [f'backend: {credentials.printable(saved.url)}']
    if (account := saved.account) is not None:
        shown = credentials.printable
        lines += [
            f'key: {shown(account.key_name)} ({shown(account.key_id)})',
            f'expires: {shown(account.expires_at or "never")}',
        ]
    return lines


def _is_set(env: Mapping[str, str], name: str) -> bool:
    return bool((env.get(name) or '').strip())


def _environment_note(env: Mapping[str, str]) -> str | None:
    """What the environment does to every other command, as the resolver (:func:`plurai._endpoints.resolve`) treats it; ``None`` when it sets nothing."""
    url, token = _is_set(env, URL_ENV), _is_set(env, TOKEN_ENV)
    if url and token:
        return (
            f'{URL_ENV} and {TOKEN_ENV} are set in the environment; the environment pair takes precedence over the saved credential, '
            'and the two are never mixed.'
        )
    if url:
        return (
            f'{URL_ENV} is set and {TOKEN_ENV} is not: other commands use the saved credential when it is for that backend, and '
            'otherwise refuse unless the backend is on this machine.'
        )
    if token:
        return (
            f'{TOKEN_ENV} is set and {URL_ENV} is not: other commands refuse, because a token is only sent to the backend it was issued for '
            f"and is not combined with the saved credential's url. Set {URL_ENV} too, or unset {TOKEN_ENV} to use the saved credential."
        )
    return None


def _status(env: Mapping[str, str]) -> int:
    path = credentials.file_path()
    print(f'credentials file: {path}')
    if note := _environment_note(env):
        print(note)
    moved = credentials.home_problem(env)
    if moved:
        print(f'saved credential not looked for: {moved}')
        return exit_codes.REFUSED
    saved = credentials.read(path)
    if saved.problem:
        print(f'saved credential not usable: {saved.problem}')
        return exit_codes.CREDENTIALS
    if not saved.token:
        print(f'no saved credential at {path}')
        print('Log in with: plurai login')
        print(_DASHBOARD)
        return exit_codes.CREDENTIALS
    print(*_account_lines(saved), sep='\n')
    expiry = _expiry(saved.account)
    if expiry != 'valid':
        what = 'has expired' if expiry == 'expired' else 'has an expiry date that cannot be read'
        print(f'The key {what}; run `plurai login` to create a new one.')
        return exit_codes.CREDENTIALS
    print(_EXPOSURE)
    print(_DASHBOARD)
    return exit_codes.OK


def _expiry(account: credentials.Account | None) -> Literal['valid', 'expired', 'unreadable']:
    """Whether the key of ``account`` can still be used. An expiry that does not parse is unknown, not valid."""
    if account is None or account.expires_at is None:
        return 'valid'
    try:
        expiry = datetime.fromisoformat(account.expires_at)
    except ValueError:
        return 'unreadable'
    return 'expired' if (expiry if expiry.tzinfo else expiry.replace(tzinfo=UTC)) <= datetime.now(UTC) else 'valid'


def _backend(env: Mapping[str, str]) -> tuple[str, environments.Environment]:
    """The api root the login saves and the environment row its host selects."""
    if not _is_set(env, URL_ENV):
        row = environments.default()
        url = row.api_root
    else:
        named = (env.get(URL_ENV) or '').strip()
        try:
            url = credentials.validate_url(diagnostic_base(named))
        except credentials.UrlRefusedError as refused:
            raise _Failed(f'{URL_ENV}: {refused}.', exit_codes.REFUSED) from None
        host = urlsplit(url).hostname or ''
        row = environments.for_host(host)
        if row is None:
            raise _Failed(
                f'{URL_ENV} names {host}, which is not a backend this SDK build signs in to; the browser login knows {environments.known_hosts()}. '
                f'For another backend, export {URL_ENV} and {TOKEN_ENV} instead of logging in.',
                exit_codes.REFUSED,
            )
    if row.unassigned():
        # The message names neither the row's missing values nor a row that cannot sign in; with no other usable row it
        # offers only the environment pair.
        others = ', '.join(f'{other.name} ({other.host})' for other in environments.ENVIRONMENTS if other != row and not other.unassigned())
        instead = f'Log in to another backend ({others}) with {URL_ENV}, or export' if others else 'Export'
        raise _Failed(
            f'this SDK release cannot log in to {row.name} yet. {instead} {URL_ENV} and {TOKEN_ENV}, with a key from the Plurai dashboard.',
            exit_codes.REFUSED,
        )
    return url, row


def _default_key_name() -> str:
    host = re.sub(r'[^a-z0-9-]+', '-', socket.gethostname().split('.')[0].lower()).strip('-')[:40] or 'host'
    return f'sdk-{host}-{datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")}'


def _body(response: Any) -> Any:  # noqa: ANN401 - whatever JSON the backend answered
    try:
        return response.json()
    except ValueError:
        return None


def _text(value: object, fallback: str) -> str:
    return value if isinstance(value, str) and value else fallback


def _keys_url(row: environments.Environment, *segments: str) -> str:
    return '/'.join([row.keys_base.rstrip('/'), 'api-keys', *(quote(segment, safe='') for segment in segments)])


def _is_header_safe(value: str) -> bool:
    """Whether ``value`` can be sent as a bearer token without breaking the request line or a header."""
    return value.isascii() and value.isprintable()


def _may_exist(name: str, key_id: str | None = None) -> str:
    """The sentence for a creation request whose result is not known: a key may exist."""
    named = f'a key named {name}' + (f' (id {credentials.printable(key_id)})' if key_id else '')
    return f'{named} may have been created; check the keys in the Plurai dashboard and revoke it if it is there.'


def _post_key(row: environments.Environment, access_token: str, name: str) -> tuple[Any, str]:  # noqa: ANN401 - the httpx response
    """The backend's answer to the creation request and the name that was tried last.

    A ``name`` that is taken (409) is retried once with a random suffix.
    """
    import httpx  # noqa: PLC0415 - the command's other paths stay on the standard library

    from plurai import _http  # noqa: PLC0415

    attempt = name
    try:
        with _http.Client(timeout=HTTP_TIMEOUT, follow_redirects=False) as client:
            for attempt in (name, f'{name}-{secrets.token_hex(3)}'):
                response = client.post(_keys_url(row), json={'name': attempt, 'expiresAt': None}, headers={'Authorization': f'Bearer {access_token}'})
                if response.status_code != 409:
                    break
    except KeyboardInterrupt:
        # The request may have reached the backend: no answer is not "no key".
        raise _Failed(f'the sign-in was cancelled while the key was being created. {_may_exist(attempt)}', exit_codes.CREDENTIALS) from None
    except (httpx.ConnectError, httpx.ConnectTimeout) as failure:
        raise _Failed(f'the backend could not be reached to create the key ({type(failure).__name__}). Nothing was saved.', exit_codes.TRANSPORT) from None
    except httpx.HTTPError as failure:
        # The request may have reached the backend: no answer is not "no key".
        raise _Failed(
            f'the connection to the backend failed while the key was being created ({type(failure).__name__}). {_may_exist(attempt)}',
            exit_codes.TRANSPORT,
        ) from None
    return response, attempt


def _refuse_status(response: Any, attempt: str) -> None:  # noqa: ANN401 - the httpx response
    """Raise for every answer to the creation request that is not a 2xx."""
    status = response.status_code
    if status == client_handshake.UPGRADE_STATUS:
        raise _Failed(client_handshake.upgrade_message_for(response), exit_codes.UPGRADE_REQUIRED)
    if status == 401:
        raise _Failed(f'the backend did not accept the sign-in (HTTP 401). {_NOT_ACCEPTED_HINT}', exit_codes.CREDENTIALS)
    if status == 403:
        raise _Failed(
            'the backend refused to create a key for this sign-in (HTTP 403). The account may lack the permission to create API keys; '
            f'ask an administrator of your organization, or export {URL_ENV} and {TOKEN_ENV} with a key made in the Plurai dashboard.',
            exit_codes.CREDENTIALS,
        )
    if status == 409:
        raise _Failed(
            f'the backend reported that the key name {attempt} is taken, again after a retry with another name (HTTP 409); run `plurai login` again.',
            exit_codes.TEMPFAIL,
        )
    if status == 429:
        raise _Failed('the backend asked to wait before creating another key (HTTP 429); run `plurai login` again in a moment.', exit_codes.TEMPFAIL)
    if not 200 <= status < 300:
        code = exit_codes.TRANSPORT if status >= 500 else exit_codes.BACKEND_REFUSED
        raise _Failed(f'the backend did not create the key (HTTP {status}). Nothing was saved.', code)


def _key_from(
    body: Any,  # noqa: ANN401 - whatever JSON the backend answered
    row: environments.Environment,
    tokens: oauth_client.Tokens,
    name: str,
) -> tuple[str, credentials.Account]:
    """``(secret, account)`` of the key a 2xx answer describes. Raises when the answer has no usable secret or id.

    An answer with an id and a secret the file cannot hold still made a key, and the SDK holds its secret: the key is
    revoked again with it.
    """
    if not isinstance(body, dict):
        raise _Failed(f'the backend answered without a usable key. {_may_exist(name)}', exit_codes.TRANSPORT)
    secret, key_id = body.get('secret'), body.get('id')
    known = key_id if isinstance(key_id, str) and key_id else None
    if isinstance(secret, str) and known and not credentials.TOKEN_PATTERN.fullmatch(secret) and _is_header_safe(secret):
        if _revoke_key(row, known, secret).outcome == 'revoked':
            raise _Failed(
                f'the backend created a key whose secret this SDK cannot store ({credentials.printable(name)}), and it has been revoked again. '
                'Nothing was saved.',
                exit_codes.TRANSPORT,
            )
    if not isinstance(secret, str) or not credentials.TOKEN_PATTERN.fullmatch(secret) or known is None:
        raise _Failed(f'the backend answered without a usable key. {_may_exist(name, known)}', exit_codes.TRANSPORT)
    key_id = known
    organization: dict[str, object] = body['organization'] if isinstance(body.get('organization'), dict) else {}
    user: dict[str, object] = body['user'] if isinstance(body.get('user'), dict) else {}
    expires_at = body.get('expiresAt')
    organization_id = _text(organization.get('id'), tokens.organization or '')
    user_id = _text(user.get('id'), tokens.subject or '')
    account = credentials.Account(
        organization_id=organization_id,
        organization_name=_text(organization.get('name'), organization_id),
        user_id=user_id,
        user_display=_text(user.get('display'), user_id),
        key_name=_text(body.get('name'), name),
        key_id=key_id,
        expires_at=expires_at if isinstance(expires_at, str) else None,
    )
    return secret, account


def _create_key(row: environments.Environment, tokens: oauth_client.Tokens, name: str) -> tuple[str, credentials.Account]:
    """``(secret, account)`` of a key created at ``row``'s backend with the access token as the bearer."""
    response, attempt = _post_key(row, tokens.access_token, name)
    try:
        _refuse_status(response, attempt)
        return _key_from(_body(response), row, tokens, attempt)
    except KeyboardInterrupt:
        # The backend answered, so a key may exist that this process has not stored or revoked yet.
        raise _Failed(f'the sign-in was cancelled while the key was being read. {_may_exist(attempt)}', exit_codes.CREDENTIALS) from None


def _unsaved_key_message(failure: credentials.CredentialsFileError, row: environments.Environment, secret: str, account: credentials.Account) -> str:
    """What to tell when the key exists but the file could not be written: the key is revoked again with itself when it can be."""
    key = credentials.printable(account.key_name)
    revocation = _revoke_key(row, account.key_id, secret)
    if revocation.outcome == 'revoked':
        return f'{failure}. The key {key} was created and has been revoked again. Nothing was saved.'
    return f'{failure}. The key {key} was created and could not be revoked again ({revocation.problem}); revoke it in the Plurai dashboard.'


def _lost_race_message(row: environments.Environment, secret: str, account: credentials.Account, path: Path) -> str:
    """What to tell when another login's key is in the file after this login wrote its own: this login's key is revoked."""
    key = credentials.printable(account.key_name)
    revocation = _revoke_key(row, account.key_id, secret)
    told = f"another plurai login saved its own key to {path} while this one ran, so its key won. This login's key {key} "
    if revocation.outcome == 'revoked':
        return told + "was revoked, and the file is the other login's."
    return told + f'could not be revoked ({revocation.problem}); revoke it in the Plurai dashboard.'


def _kept_key_message(cause: str, account: credentials.Account, path: Path) -> str:
    return f'{cause} after the key {credentials.printable(account.key_name)} was saved to {path}. The key stays; `plurai login --status` shows it.'


def _interrupted_save(
    cause: str, row: environments.Environment, secret: str, account: credentials.Account, path: Path, *, previous: credentials.Credential
) -> _Failed:
    """The failure for an interrupt between the key's creation and the confirmed save.

    A file that already holds the key keeps it (:class:`_CancelledAfterSave`, which still retires ``previous``): revoking
    it would leave the file with a dead key. Otherwise the key is revoked again with itself.
    """
    saved = credentials.read(path)
    if not saved.problem and saved.token == secret:
        return _CancelledAfterSave(_kept_key_message(cause, account, path), previous)
    unsaved = credentials.CredentialsFileError(f'{cause} while the key was being saved')
    return _Failed(_unsaved_key_message(unsaved, row, secret, account), exit_codes.CREDENTIALS)


def _store_key(
    flow: oauth_client.Authorization, url: str, row: environments.Environment, secret: str, account: credentials.Account, *, path: Path
) -> credentials.Credential:
    """Write the key, confirm the file holds it, and return what the file held just before the write: the key to retire.

    Any other outcome revokes the key again (see :func:`_interrupted_save`) and raises. The browser gets its failure page
    before the revocation, which can take a while.

    The key to retire is read right before the write, not when the login starts: another login that saves its key while
    this one waits for the browser is then retired by this one, and this login's key by any login that writes after it. A
    lost read-back still carries ``previous``: a winner that read the file after this write retires this login's key, not
    the key this login replaced, so nobody else can retire that one.
    """
    previous = credentials.Credential()
    try:
        previous = credentials.read(path)
        credentials.write(url, secret, path, account=account)
    except credentials.FileReplacedError:
        flow.respond(succeeded=False)
        raise _FailedAfterWrite(_lost_race_message(row, secret, account, path), exit_codes.REFUSED, previous) from None
    except credentials.CredentialsFileError as failure:
        flow.respond(succeeded=False)
        raise _Failed(_unsaved_key_message(failure, row, secret, account), exit_codes.REFUSED) from None
    except BaseException as interrupted:
        flow.respond(succeeded=False)
        cancelled = isinstance(interrupted, KeyboardInterrupt)
        cause = 'the sign-in was cancelled' if cancelled else 'the login failed unexpectedly'
        failure = _interrupted_save(cause, row, secret, account, path, previous=previous)
        if cancelled:
            raise failure from None
        print(f'plurai login: {failure}', file=sys.stderr)
        raise
    return previous


def _end_sign_in(flow: oauth_client.Authorization, tokens: oauth_client.Tokens) -> None:
    """Revoke the refresh token; a failure is told, not raised, as the sign-in's outcome is already settled."""
    from plurai.login import oauth_client  # noqa: PLC0415

    if tokens.refresh_token is None:
        return
    try:
        flow.revoke(tokens.refresh_token, 'refresh_token')
    except oauth_client.RevocationError as failure:
        print(f'plurai login: {failure}; the sign-in session was not ended, and nothing of it is stored on this machine.', file=sys.stderr)
    except Exception as failure:  # an unexpected fault of the HTTP stack must not replace the outcome that is already settled
        print(f'plurai login: the sign-in session was not ended ({type(failure).__name__}), and nothing of it is stored on this machine.', file=sys.stderr)


def _saved_line(url: str, path: Path) -> str:
    protection = 'access limited to your user' if credentials.is_windows() else 'mode 0600'
    return f'Saved the key for {url} in {path} ({protection}). The key is not shown.'


def _sign_in(url: str, row: environments.Environment, name: str, env: Mapping[str, str]) -> credentials.Credential:
    """Sign in, create the key and save it; the return is what the file held just before the save (see :func:`_store_key`)."""
    from plurai.login import oauth_client  # noqa: PLC0415 - Authlib loads only when a sign-in runs

    path = credentials.file_path()
    has_display = _can_open_browser(env)
    kept: credentials.Account | None = None
    previous = credentials.Credential()
    try:
        with oauth_client.Authorization(
            issuer=row.issuer,
            client_id=row.client_id,
            scopes=SCOPES,
            open_browser=open_browser if has_display else _no_browser,
            success_url=row.connected_url,
        ) as flow:
            try:
                print(
                    'Opening the browser to sign in. If it does not open, visit this url:'
                    if has_display
                    else 'No screen is available here. Open this url in a browser:'
                )
                print(flow.authorization_url, flush=True)
                tokens = flow.wait()
            except KeyboardInterrupt:
                raise _Failed('the sign-in was cancelled. Nothing was saved.', exit_codes.CREDENTIALS) from None
            # From here the tokens exist: whatever happens next, the refresh token is revoked and both are dropped.
            try:
                if tokens.organization is None:
                    raise _Failed(f'the sign-in carries no organization. {_ORGANIZATION_HINT}', exit_codes.CREDENTIALS)
                secret, account = _create_key(row, tokens, name)
                previous = _store_key(flow, url, row, secret, account, path=path)
                kept = account
                flow.respond(succeeded=True)
            except BaseException:
                flow.respond(succeeded=False)
                raise
            finally:
                _end_sign_in(flow, tokens)
                del tokens
    except KeyboardInterrupt:
        if kept is None:
            raise
        raise _CancelledAfterSave(_kept_key_message('the sign-in was cancelled', kept, path), previous) from None
    except oauth_client.ConsentDeniedError:
        raise _Failed('the sign-in was declined at the consent screen. Nothing was saved.', exit_codes.CREDENTIALS) from None
    except oauth_client.CallbackTimeoutError as failure:
        raise _Failed(f'{failure}. Nothing was saved; run `plurai login` again to retry.', exit_codes.TEMPFAIL) from None
    except oauth_client.ListenerError as failure:
        raise _Failed(f'{failure}. Nothing was saved.', exit_codes.REFUSED) from None
    except oauth_client.IssuerMismatchError as failure:
        raise _Failed(f'{failure}; the sign-in was stopped. Nothing was saved.', exit_codes.REFUSED) from None
    except oauth_client.DiscoveryError as failure:
        raise _Failed(f'the sign-in server could not be used: {failure}. Nothing was saved.', exit_codes.TRANSPORT) from None
    except oauth_client.TokenEndpointError as failure:
        if failure.status == 429:
            code = exit_codes.TEMPFAIL
        elif failure.status is None or failure.status >= 500 or 200 <= failure.status < 300:  # a 2xx with no usable token is an unusable answer
            code = exit_codes.TRANSPORT
        else:
            code = exit_codes.CREDENTIALS
        raise _Failed(f'{failure}. Nothing was saved.', code) from None
    except oauth_client.AuthorizationError as failure:
        code = {'temporarily_unavailable': exit_codes.TEMPFAIL, 'server_error': exit_codes.TRANSPORT}.get(failure.error or '', exit_codes.CREDENTIALS)
        raise _Failed(f'{failure}. Nothing was saved.', code) from None
    except oauth_client.OAuthClientError as failure:
        raise _Failed(f'{failure}. Nothing was saved.', exit_codes.CREDENTIALS) from None
    print(_saved_line(url, path))
    return previous


def run(args: argparse.Namespace, *, env: Mapping[str, str] | None = None) -> int:
    environment = os.environ if env is None else env
    if args.status:
        return _status(environment)
    try:
        return _login(environment)
    except KeyboardInterrupt:
        # The waits and the key's creation and save answer an interrupt themselves; this is any other point of the login.
        return _refuse('the sign-in was cancelled. `plurai login --status` shows whether a key was saved.', exit_codes.CREDENTIALS)


def _login(environment: Mapping[str, str]) -> int:
    if _is_set(environment, TOKEN_ENV):
        return _refuse(
            f'{TOKEN_ENV} is set, and every client uses the environment pair over a saved login, so a login would save a key nothing reads. '
            f'Unset {TOKEN_ENV} to log in, or keep using the pair.',
            exit_codes.USAGE,
        )
    name = _default_key_name()
    if moved := credentials.home_problem(environment):
        return _refuse(f'{moved}. Nothing was saved.')
    path = credentials.file_path()
    try:
        url, row = _backend(environment)
        saved = credentials.read(path)
        same_backend = not saved.problem and bool(saved.token) and diagnostic_base(saved.url) == diagnostic_base(url)
        if same_backend and _expiry(saved.account) == 'valid':
            print(f'Already logged in; no login is needed. The saved key in {path} is for:')
            print(*_account_lines(saved), sep='\n')
            print('If the key no longer works, run `plurai logout`, then `plurai login`.')
            return exit_codes.OK
        try:
            credentials.check_writable(path)
        except credentials.CredentialsFileError as failure:
            raise _Failed(f'{failure}. Nothing was saved.', exit_codes.REFUSED) from None
        previous = _sign_in(url, row, name, environment)
    except _FailedAfterWrite as failed:
        code = _refuse(str(failed), failed.code)
        _retire_previous_or_name_it(failed.previous, path)
        return code
    except _Failed as failure:
        return _refuse(str(failure), failure.code)
    if not _retire_previous_or_name_it(previous, path):
        return exit_codes.CREDENTIALS
    print('Logged in to Plurai.')
    print(_EXPOSURE)
    print(_DASHBOARD)
    return exit_codes.OK


def main(argv: Sequence[str]) -> int:
    parser = _Parser(prog='plurai login', description=(__doc__ or '').split('\n\n')[0])
    add_arguments(parser)
    return run(parser.parse_args(list(argv)))


@dataclass(frozen=True)
class _Revocation:
    """What became of a key at the backend: ``revoked`` (confirmed), ``not_accepted`` (the backend no longer knows the key) or ``failed``.

    ``code`` is the exit status a failure maps to; a failure is :attr:`retryable` when running the command again can fix it
    (the backend is unreachable, 429, 5xx, an outdated client).
    """

    outcome: Literal['revoked', 'not_accepted', 'failed'] = 'revoked'
    problem: str = ''
    code: int = exit_codes.OK

    @property
    def retryable(self) -> bool:
        return self.outcome == 'failed' and self.code in (exit_codes.TEMPFAIL, exit_codes.UPGRADE_REQUIRED)


def _revoke_key(row: environments.Environment, key_id: str, bearer: str) -> _Revocation:
    """Revoke key ``key_id`` with ``bearer`` as the key. The token goes only to the key routes of the SDK's own table, never to a file's url."""
    import httpx  # noqa: PLC0415

    from plurai import _http  # noqa: PLC0415

    try:
        with _http.Client(timeout=HTTP_TIMEOUT, follow_redirects=False) as client:
            response = client.patch(_keys_url(row, key_id), json={'revoke': True}, headers={'Authorization': f'Bearer {bearer}'})
    except httpx.HTTPError as failure:
        return _Revocation('failed', f'the backend could not be reached ({type(failure).__name__})', exit_codes.TEMPFAIL)
    status = response.status_code
    if 200 <= status < 300:
        return _Revocation()
    if status in (401, 404):  # the key authenticates its own revocation, so a 401 or 404 can also mean a wrong route: never "revoked"
        return _Revocation('not_accepted', f'the backend no longer accepts the key (HTTP {status})')
    if status == client_handshake.UPGRADE_STATUS:
        return _Revocation('failed', client_handshake.upgrade_message_for(response), exit_codes.UPGRADE_REQUIRED)
    if status == 429 or status >= 500:
        return _Revocation('failed', f'the backend answered HTTP {status}', exit_codes.TEMPFAIL)
    return _Revocation('failed', f'the backend answered HTTP {status}', exit_codes.CREDENTIALS)


def _revoke_saved_key(saved: credentials.Credential) -> _Revocation:
    account = saved.account
    if account is None:
        return _Revocation('failed', 'the file does not say which key it holds', exit_codes.CREDENTIALS)
    row = environments.for_host(urlsplit(saved.url).hostname or '')
    if row is None or row.unassigned():
        return _Revocation('failed', f'this SDK has no key routes for {credentials.printable(saved.url)}', exit_codes.CREDENTIALS)
    return _revoke_key(row, account.key_id, saved.token)


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


def _key_label(saved: credentials.Credential) -> str:
    """The name of the saved key for a message, or the words for a file that did not record it."""
    return credentials.printable(saved.account.key_name) if saved.account is not None else 'a key whose name this file did not record'


def _retire_previous(saved: credentials.Credential, path: Path) -> None:
    """After a login overwrote ``saved``: revoke its key best effort, or say that it was not and where to revoke it."""
    if saved.problem:
        print(
            f'The previous file at {path} could not be used ({credentials.printable(saved.problem)}). '
            'Its key was not revoked; if it may have been copied, revoke it in the dashboard.',
            file=sys.stderr,
        )
        return
    if not saved.token:
        return
    revocation = _revoke_saved_key(saved)
    if revocation.outcome == 'revoked':
        print(f'Revoked the previous key ({_key_label(saved)}).')
    elif revocation.outcome == 'not_accepted':
        print(f'{_sentence(revocation.problem)}: the previous key ({_key_label(saved)}) was not revoked by this command. {_REVOKE_IN_DASHBOARD}')
    else:
        print(
            f'plurai login: the previous key was not revoked: {revocation.problem}. '
            f'The key ({_key_label(saved)}) still works for anyone who has a copy. {_REVOKE_IN_DASHBOARD}',
            file=sys.stderr,
        )


def _retire_previous_or_name_it(saved: credentials.Credential, path: Path) -> bool:
    """:func:`_retire_previous`; on a Ctrl-C during it, name the key that may still work and return False."""
    try:
        _retire_previous(saved, path)
    except KeyboardInterrupt:
        _refuse(
            f'the sign-in was cancelled while the previous key ({_key_label(saved)}) was being revoked. '
            f'It may still work for anyone who has a copy. {_REVOKE_IN_DASHBOARD}',
        )
        return False
    return True


def run_logout(*, env: Mapping[str, str] | None = None) -> int:
    environment = os.environ if env is None else env
    if moved := credentials.home_problem(environment):
        print(f'plurai logout: {moved}. Nothing was changed.', file=sys.stderr)
        return exit_codes.REFUSED
    path = credentials.file_path()
    saved = credentials.read(path)
    if saved.problem:
        print(f'plurai logout: the saved credential is not usable: {saved.problem}. Nothing was changed.', file=sys.stderr)
        return exit_codes.CREDENTIALS
    if not saved.token:
        print(f'No saved credential at {path}; nothing to log out of.')
        return exit_codes.OK
    revocation = _revoke_saved_key(saved)
    if revocation.retryable:
        print(
            f'plurai logout: the key was not revoked: {revocation.problem}. {path} was kept; run `plurai logout` again.',
            file=sys.stderr,
        )
        return revocation.code
    try:
        credentials.delete(path)
    except credentials.CredentialsFileError as failure:
        print(
            f'plurai logout: {failure}. The key was {"revoked" if revocation.outcome == "revoked" else "not revoked"}; remove the file yourself.',
            file=sys.stderr,
        )
        return exit_codes.REFUSED
    if _is_set(environment, TOKEN_ENV):
        print(f'{TOKEN_ENV} is still set in the environment; clients keep using it until you unset it.')
    if revocation.outcome == 'failed':
        print(
            f'plurai logout: removed {path}, but the key was not revoked: {revocation.problem}. '
            f'The key ({_key_label(saved)}) still works for anyone who has a copy. {_REVOKE_IN_DASHBOARD}',
            file=sys.stderr,
        )
        return revocation.code
    if revocation.outcome == 'not_accepted':
        print(
            f'Removed {path}. {_sentence(revocation.problem)}, so this command did not revoke it. '
            'If a copy of the key still works, revoke it in the Plurai dashboard.'
        )
        return exit_codes.OK
    print(f'Revoked the key and removed {path}.')
    return exit_codes.OK


def main_logout(argv: Sequence[str]) -> int:
    parser = _Parser(prog='plurai logout', description='Revoke the saved key at the backend and remove the credentials file.')
    parser.parse_args(list(argv))
    return run_logout()
