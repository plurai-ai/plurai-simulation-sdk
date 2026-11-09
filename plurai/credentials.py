"""The credentials file: the one file on a machine that names the Plurai backend and holds the token for it.

This module is the specification every Plurai client conforms to; `tests/fixtures/credentials_parity.json`
holds the cases the other clients assert too.

* Path: ``~/.plurai/credentials.json`` in the home directory the account database names. A ``HOME`` that names
  another directory is not followed: a command a model runs sets its own environment, and where the credential
  lives must not move with it. Windows has no account database; there the home is the profile folder the operating
  system names for the user (``SHGetKnownFolderPath``), never ``HOME``, ``USERPROFILE``, ``HOMEDRIVE`` or
  ``HOMEPATH``. A profile on a network path or a drive that is not fixed (a profile that is a link is followed to its
  target, which must be a fixed local folder), or a ``.plurai`` folder that resolves out of the profile, is not used (``plurai._winsec``, whose cases are in
  ``tests/fixtures/credentials_windows_vectors.json``).
* Content: a JSON object ``{"url": ..., "token": ...}``, written as ``json.dumps(..., indent=2)`` plus a newline.
  The browser login adds ``account`` (:class:`Account`): non-secret display data about the key, which a reader that
  does not use it ignores, and which this module ignores when it is malformed. A file with a ``version`` or ``origin``
  key is another format and is not read. The token is sent only to requests under the url's origin (the resolver
  enforces that, not this module).
* Safety: the directory is mode 0700, the file mode 0600, both owned by the user and neither a symlink. The file
  is written atomically and read through the descriptor that was checked. On Windows the same is done with access
  lists: the folder and the file are limited to the current user before the key is written, neither may be a symlink
  or any other reparse point, and the descriptor opened must be the file that was checked. A read also refuses a file
  or folder that neither the user nor the Administrators group owns (an elevated process creates files for the latter),
  or whose access list allows any trustee but the user, SYSTEM and Administrators (``GetSecurityInfo``, on the open
  descriptor's own handle for the file).
* Allowed urls: ``https`` under ``*.plurai.ai``, or loopback over either scheme. Every other origin reaches a
  client only through ``PLURAI_URL`` in the environment that launched it, so a login a model runs cannot repoint
  later sessions at a host of its choosing. A url carries no userinfo, query, fragment, escape or backslash, no port 0,
  and a path of plain segments.
* Token: one run of letters, digits and ``. _ ~ + / = -``, :data:`MIN_TOKEN_LENGTH` to 4096 characters. A shorter value
  is refused because the SDK would not redact it from what it writes.

Stdlib only: `plurai login` stays off the heavy import path.
"""

from __future__ import annotations

import contextlib
import errno
import ipaddress
import json
import ntpath
import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from plurai import _private_files, _winsec
from plurai._platform import is_reparse_point, is_windows

#: Hosts under this suffix may be named by the file. A glob ``*.plurai.ai``: the bare domain is not under it.
ALLOWED_SUFFIX = '.plurai.ai'
ALLOWED_ORIGINS_TEXT = '*.plurai.ai or loopback'
#: The shortest value the SDK redacts as a credential (:func:`plurai.host_effects.redact_cli_credentials`) and so the
#: shortest token it accepts: a token it cannot redact would reach the transcripts and logs whole.
MIN_TOKEN_LENGTH = 8
TOKEN_PATTERN = re.compile(rf'[A-Za-z0-9._~+/=-]{{{MIN_TOKEN_LENGTH},4096}}')
TOKEN_SHAPE = f'it must be one run of letters, digits and . _ ~ + / = - (no spaces), {MIN_TOKEN_LENGTH} to 4096 characters'  # noqa: S105
_MAX_BYTES = 64 * 1024
_PATH = re.compile(r'(?:/[A-Za-z0-9._~-]+)*/?')
_LOGIN_AGAIN = 'log in again with `plurai login`'
_FORMAT_MESSAGE = f'is not a Plurai credentials file; {_LOGIN_AGAIN}'


class UrlRefusedError(ValueError):
    """A url the credentials file may not hold. The message never repeats the url, which may carry a secret."""


class CredentialsFileError(RuntimeError):
    """The file or its directory cannot be written safely."""


class FileReplacedError(CredentialsFileError):
    """The file was written, and what it holds when read back is another writer's credential: that writer won."""


@dataclass(frozen=True)
class Account:
    """Who the saved key belongs to and which key it is, as the backend named them when it created the key."""

    organization_id: str
    organization_name: str
    user_id: str
    user_display: str
    key_name: str
    key_id: str
    #: An ISO 8601 instant, or ``None`` for a key that does not expire.
    expires_at: str | None

    def to_json(self) -> dict[str, object]:
        return {
            'organization': {'id': self.organization_id, 'name': self.organization_name},
            'user': {'id': self.user_id, 'display': self.user_display},
            'key_name': self.key_name,
            'key_id': self.key_id,
            'expires_at': self.expires_at,
        }

    @classmethod
    def from_json(cls, value: object) -> Account | None:
        """The account ``value`` describes, or ``None`` when it is not the shape :meth:`to_json` writes."""
        if not isinstance(value, dict):
            return None
        organization, user, expires = value.get('organization'), value.get('user'), value.get('expires_at')
        if not isinstance(organization, dict) or not isinstance(user, dict) or not (expires is None or isinstance(expires, str)):
            return None
        fields = (organization.get('id'), organization.get('name'), user.get('id'), user.get('display'), value.get('key_name'), value.get('key_id'))
        if not all(isinstance(item, str) for item in fields):
            return None
        return cls(*fields, expires_at=expires)  # type: ignore[arg-type]  # every field was checked to be a str


@dataclass(frozen=True)
class Credential:
    """What the file holds. ``problem`` says why a file that exists was not used; a missing file is no credential and no problem."""

    url: str = ''
    token: str = field(default='', repr=False)
    problem: str = ''
    account: Account | None = None


#: Said with every reason a profile cannot be used: the backend can still be named in the environment.
_ENVIRONMENT_HINT = '; a backend named in the environment (PLURAI_URL and PLURAI_TOKEN) is still used'


def printable(text: str) -> str:
    """``text`` with every character that is not printable replaced by ``?``, for a message that names a path or a backend-supplied name.

    That covers the C0 and C1 controls, the line separators and the format characters such as the bidirectional overrides; letters
    of any script stay. A profile folder, the target of a link or an account name comes from the system, the file system or the backend, and
    it may hold a newline or an escape sequence that would forge a line of the message or drive the terminal it is printed on.
    """
    return ''.join(char if char.isprintable() else '?' for char in text)


def windows_profile(real: Callable[[str], str] = os.path.realpath) -> tuple[str, str]:
    """``(the user's profile folder, why it may not hold the credential)`` on Windows; the reason is ``''`` when it may.

    The folder is the operating system's answer, not an environment variable's. The reason is a sentence naming the folder
    and, for a link, its target (control characters replaced); the folder returned is the system's answer as it is.
    """
    try:
        profile = _winsec.profile_directory()
        drive = _winsec.drive_type(_winsec.local_root(profile))
        resolved = real(profile)
        # A `\\?\Volume{GUID}\` root names the volume of a profile container mounted without a drive letter; it is accepted
        # on the assumption that the profile lies on that volume.
        resolved_root = _winsec.local_root(resolved)
        resolved_drive = _winsec.drive_type(resolved_root) if resolved_root else 0
    except OSError as exc:
        return '', f'the user profile cannot be determined ({exc}), so the saved credential is not looked for{_ENVIRONMENT_HINT}'
    code = _winsec.profile_problem(profile, drive, resolved, resolved_drive)
    if code:
        target = '' if resolved.lower() == profile.lower() else f' (it resolves to {printable(resolved)})'
        return profile, _winsec.message(code, printable(profile)) + target + _ENVIRONMENT_HINT
    folder = ntpath.join(profile, '.plurai')
    resolved_folder = real(folder)
    if code := _winsec.folder_problem(resolved, resolved_folder):
        return (
            profile,
            f'{_winsec.message(code, printable(folder))} (it resolves to {printable(resolved_folder)}); remove the link or move it away{_ENVIRONMENT_HINT}',
        )
    return profile, ''


def windows_home() -> Path:
    """The home on Windows: the profile folder the operating system names (``Path('~')``, under which nothing is found, when there is none)."""
    profile, _ = windows_profile()
    return Path(profile) if profile else Path('~')


def account_home() -> Path:
    """The user's home directory as the account database has it; ``~``'s expansion for an account with no entry."""
    if is_windows():
        return windows_home()
    try:
        import pwd

        home = pwd.getpwuid(os.getuid()).pw_dir
    except (ImportError, KeyError):
        home = ''
    return Path(home) if home else Path('~').expanduser()


def home_problem(env: Mapping[str, str] | None = None, *, is_win: bool | None = None) -> str:
    """Why the file is not looked for under ``env``, or ``''``: a ``HOME`` that is set and names another directory than the account's.

    On Windows ``HOME`` is not read; the reason is the profile folder's, if it may not be used.
    """
    if is_windows() if is_win is None else is_win:
        return windows_profile()[1]
    named = (os.environ if env is None else env).get('HOME')
    if not named:
        return ''
    actual = account_home()
    try:
        if Path(named).samefile(actual):
            return ''
    except OSError:
        pass
    return (
        f'HOME is {named}, not the account home {actual}, so the saved credential is not looked for there; '
        'run this with your own HOME, or export PLURAI_URL and PLURAI_TOKEN'
    )


def file_path() -> Path:
    return account_home() / '.plurai' / 'credentials.json'


def is_loopback_host(host: str) -> bool:
    """True for ``localhost`` and a literal loopback address (``127.0.0.0/8``, ``::1``); any other name is remote."""
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_url(url: str) -> str:
    """``url`` as the file stores it (surrounding whitespace removed), or ``UrlRefusedError`` saying why not."""
    text = url.strip()
    if not text or '\\' in text or any(ord(char) <= 0x20 or ord(char) == 0x7F for char in text):
        raise UrlRefusedError('the url must be one absolute https url, with no whitespace or backslash')
    if any(char in text for char in '?#%@'):
        raise UrlRefusedError('the url must not carry userinfo, a query, a fragment or an escape, which two parsers could read differently')
    try:
        parts = urlsplit(text)
        scheme, host, port = parts.scheme.lower(), parts.hostname or '', parts.port
    except ValueError:
        raise UrlRefusedError('the value is not a url') from None
    if scheme not in ('http', 'https') or not host or not host.isascii():
        raise UrlRefusedError('the url must be an absolute https url with a host')
    if port == 0:
        raise UrlRefusedError('the url names port 0')
    if not _PATH.fullmatch(parts.path) or any(segment in ('.', '..') for segment in parts.path.split('/')):
        raise UrlRefusedError("the url has a path that is not a plain run of segments (no '..', '//' or escapes)")
    if is_loopback_host(host):
        return text
    if scheme != 'https':
        raise UrlRefusedError(f'{host} must be reached over https: only loopback may use http')
    if not host.endswith(ALLOWED_SUFFIX) or not all(host.split('.')):
        raise UrlRefusedError(
            f'{host} is not under {ALLOWED_SUFFIX.replace(".", "*.", 1)} and is not loopback, so no client would use a token saved for it. '
            f'Allowed origins: {ALLOWED_ORIGINS_TEXT}. To use another backend, export PLURAI_URL and PLURAI_TOKEN in the environment that launches the client'
        )
    return text


def _open_private_directory(path: Path, uid: int) -> int:
    """A descriptor for ``path`` as a directory only the user can use, created when missing."""
    try:
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            path.mkdir(mode=0o700)
            info = os.lstat(path)
    except OSError as exc:
        raise CredentialsFileError(f'{path} cannot be inspected or created ({exc.strerror})') from exc
    if stat.S_ISLNK(info.st_mode):
        raise CredentialsFileError(f'{path} is a symlink, which is not followed; remove it or move it away')
    if not stat.S_ISDIR(info.st_mode):
        raise CredentialsFileError(f'{path} is not a directory')
    if info.st_uid != uid:
        raise CredentialsFileError(f'{path} is not owned by you')
    if stat.S_IMODE(info.st_mode) & 0o022:
        raise CredentialsFileError(f'{path} is writable by other users; run `chmod 700 {path}` and check what it holds')
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise CredentialsFileError(f'{path} cannot be opened ({exc.strerror})') from exc
    try:
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o700:
            os.fchmod(descriptor, 0o700)
    except OSError as exc:
        os.close(descriptor)
        raise CredentialsFileError(f'{path} cannot be made private ({exc.strerror})') from exc
    return descriptor


def _refuse_symlink(target: Path, directory: int) -> None:
    try:
        existing = os.lstat(target.name, dir_fd=directory)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(existing.st_mode):
        raise CredentialsFileError(f'{target} is a symlink, which is not followed; remove it first')


def _require_windows_folder(folder: Path, info: os.stat_result) -> None:
    """Raise unless ``folder``, as ``info`` describes it, is a real directory: a reparse point is never followed."""
    if is_reparse_point(info):
        raise CredentialsFileError(f'{folder} is a symlink or junction, which is not followed; remove it or move it away')
    if not stat.S_ISDIR(info.st_mode):
        raise CredentialsFileError(f'{folder} is not a directory')


def _restrict_windows_folder(folder: Path) -> None:
    """Windows: ``folder`` made the user's alone every time (not only when created), never a reparse point."""
    try:
        try:
            info = os.lstat(folder)
        except FileNotFoundError:
            folder.mkdir()
            info = os.lstat(folder)
    except OSError as exc:
        raise CredentialsFileError(f'{folder} cannot be inspected or created ({exc.strerror})') from exc
    _require_windows_folder(folder, info)
    # Every write: a folder that already exists may be another account's, or open to others, and a new file inherits its list.
    _private_files.restrict_directory(folder)


def check_writable(path: Path, *, is_win: bool | None = None) -> None:
    """Raise ``CredentialsFileError`` when :func:`write` would refuse ``path``; creates the private directory when missing.

    For a command that must not start work whose result it could not save.
    """
    if is_windows() if is_win is None else is_win:
        try:
            _restrict_windows_folder(path.parent)
        except OSError as error:  # cutting the access list failed
            raise CredentialsFileError(f'{path} cannot be written ({error.strerror or error})') from error
        try:
            existing = os.lstat(path)
        except FileNotFoundError:
            return
        except OSError as error:
            raise CredentialsFileError(f'{path} cannot be written ({error.strerror or error})') from error
        if is_reparse_point(existing):
            raise CredentialsFileError(f'{path} is a symlink or junction, which is not followed; remove it first')
        return
    directory = _open_private_directory(path.parent, os.getuid())
    try:
        _refuse_symlink(path, directory)
    finally:
        os.close(directory)


def delete(path: Path, *, is_win: bool | None = None) -> bool:
    """Remove the credentials file; ``True`` when one was removed, ``False`` when there was none. Raises ``CredentialsFileError``.

    The directory is checked the way :func:`write` checks it and is never created. The file itself is unlinked by name
    inside that directory, so a symlink in its place is removed and never followed. On Windows the folder may not be a
    reparse point and is left as it is; a link in the file's place is removed, never its target.
    """
    if not os.path.lexists(path.parent):
        return False
    if is_windows() if is_win is None else is_win:
        return _delete_windows(path)
    directory = _open_private_directory(path.parent, os.getuid())
    try:
        os.unlink(path.name, dir_fd=directory)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise CredentialsFileError(f'{path} cannot be removed ({error.strerror or type(error).__name__})') from error
    finally:
        os.close(directory)
    return True


def _delete_windows(target: Path) -> bool:
    try:
        _require_windows_folder(target.parent, os.lstat(target.parent))
        target.unlink()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise CredentialsFileError(f'{target} cannot be removed ({error.strerror or type(error).__name__})') from error
    return True


def write(url: str, token: str, path: Path | None = None, *, account: Account | None = None, is_win: bool | None = None) -> None:
    """Save ``{url, token}`` (and ``account`` when given) atomically with mode 0600 in a directory of mode 0700, then read it back.

    ``url`` and ``token`` must already have passed ``validate_url`` and ``TOKEN_PATTERN``. Raises ``CredentialsFileError``,
    and its subclass ``FileReplacedError`` when the read-back finds another writer's credential.
    On Windows the modes are an access list for the current user alone (see the module docstring).
    """
    target = file_path() if path is None else path
    document: dict[str, object] = {'url': url, 'token': token}
    if account is not None:
        document['account'] = account.to_json()
    data = (json.dumps(document, indent=2) + '\n').encode('utf-8')
    windows = is_windows() if is_win is None else is_win
    if windows:
        if path is None and (unusable := home_problem(is_win=True)):
            raise CredentialsFileError(f'{unusable}; nothing was saved')
        _write_windows(target, data)
    else:
        _write_posix(target, data)
    saved = read(target, is_win=windows)
    if saved.problem:
        raise CredentialsFileError(f'the file was written but cannot be read back: {saved.problem}')
    if (saved.url, saved.token, saved.account) != (url, token, account):
        raise FileReplacedError('the file was written, and another writer replaced it before it was read back')


def _write_windows(target: Path, data: bytes) -> None:
    try:
        _restrict_windows_folder(target.parent)
        _private_files.write_atomic(target, data)
    except OSError as error:
        raise CredentialsFileError(f'{target} cannot be written ({error.strerror or error}); nothing was saved') from error


def _write_posix(target: Path, data: bytes) -> None:
    directory = _open_private_directory(target.parent, os.getuid())
    try:
        _refuse_symlink(target, directory)
        temporary = f'.{target.name}.{os.getpid()}.tmp'
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=directory)  # a leftover of an earlier process with this pid
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        try:
            with os.fdopen(descriptor, 'wb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target.name, src_dir_fd=directory, dst_dir_fd=directory)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary, dir_fd=directory)
            raise
    except OSError as error:
        raise CredentialsFileError(f'{target} cannot be written ({error.strerror or type(error).__name__}); nothing was saved') from error
    finally:
        os.close(directory)


def _fix_or_delete(folder: Path) -> str:
    return f'; fix its owner and access list, or delete {folder} and run `plurai login` again'


def _security_problem(path: Path, folder: Path, user: str, read_security: Callable[[], _winsec.FileSecurity]) -> str:
    """Why the owner or the access list of ``path`` (the file or its folder) is not the user's alone, or ``''``; the reason names what to fix."""
    try:
        security = read_security()
    except OSError as exc:
        return f'the owner and access list of {path} cannot be read ({exc})'
    code = _winsec.access_problem(security.owner, security.allowed, security.dacl_present, user)
    return f'{_winsec.message(code, path)}{_fix_or_delete(folder)}' if code else ''


def _read_windows(target: Path) -> bytes | Credential:
    """The bytes of the file at ``target`` on Windows, or the ``Credential`` that says why not.

    Neither the file nor its folder may be a reparse point, the name may not be an alternate data stream, and the
    descriptor opened must be the file ``lstat`` judged (same volume and file index): the check that stands in for
    ``O_NOFOLLOW``. The folder is checked on a handle of its own before the file is opened, and the file on the open
    descriptor's own handle: the object checked is the object read. Both must be owned by the user or the Administrators
    group and their access lists must allow no one but the user, SYSTEM and Administrators.
    """
    if ':' in target.name:
        return Credential(problem=f'{target} names an alternate data stream, which is not a credentials file')
    try:
        if is_reparse_point(os.lstat(target.parent)):
            return Credential(problem=f'{target.parent} is a symlink or junction, which is not followed; {_LOGIN_AGAIN}')
        judged = os.lstat(target)
    except FileNotFoundError:
        return Credential()
    except OSError as exc:
        return Credential(problem=f'{target} cannot be inspected ({exc.strerror})')
    if is_reparse_point(judged):
        return Credential(problem=f'{target} is a symlink or junction, which is not followed; {_LOGIN_AGAIN}')
    if not stat.S_ISREG(judged.st_mode):
        return Credential(problem=f'{target} is not a regular file')
    try:
        user = _winsec.user_sid()
    except OSError as exc:
        return Credential(problem=f'the current user cannot be identified ({exc})')
    if problem := _security_problem(target.parent, target.parent, user, lambda: _winsec.file_security(target.parent)):
        return Credential(problem=problem)
    try:
        descriptor = os.open(target, os.O_RDONLY | getattr(os, 'O_BINARY', 0))
    except FileNotFoundError:
        return Credential()
    except OSError as exc:
        return Credential(problem=f'{target} cannot be opened ({exc.strerror})')
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != (judged.st_dev, judged.st_ino):
            return Credential(problem=f'{target} changed while it was being opened; try again')
        if problem := _security_problem(target, target.parent, user, lambda: _winsec.descriptor_security(descriptor)):
            return Credential(problem=problem)
        return os.read(descriptor, _MAX_BYTES + 1)
    finally:
        os.close(descriptor)


def read(path: Path | None = None, *, uid: int | None = None, is_win: bool | None = None) -> Credential:
    """The credential at ``path`` (default ``file_path()``), or why it was not used."""
    target = file_path() if path is None else path
    if is_windows() if is_win is None else is_win:
        if path is None and (unusable := home_problem(is_win=True)):
            return Credential(problem=unusable)
        found = _read_windows(target)
        return found if isinstance(found, Credential) else _parse(target, found)
    me = os.getuid() if uid is None else uid
    try:
        if stat.S_ISLNK(os.lstat(target.parent).st_mode):
            return Credential(problem=f'{target.parent} is a symlink, which is not followed; {_LOGIN_AGAIN}')
    except FileNotFoundError:
        return Credential()
    except OSError as exc:
        return Credential(problem=f'{target.parent} cannot be inspected ({exc.strerror})')
    try:
        descriptor = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, 'O_NONBLOCK', 0))
    except FileNotFoundError:
        return Credential()
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            return Credential(problem=f'{target} is a symlink, which is not followed; {_LOGIN_AGAIN}')
        return Credential(problem=f'{target} cannot be opened ({exc.strerror})')
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            return Credential(problem=f'{target} is not a regular file')
        if info.st_uid != me:
            return Credential(problem=f'{target} is not owned by the user running this command')
        if stat.S_IMODE(info.st_mode) & 0o077:
            mode = stat.S_IMODE(info.st_mode)
            return Credential(problem=f'{target} permissions let other users read it (mode {mode:o}); run `chmod 600` on it or {_LOGIN_AGAIN}')
        folder = target.parent.stat()
        if folder.st_uid != me or stat.S_IMODE(folder.st_mode) & 0o022:
            return Credential(problem=f'{target.parent} must belong to the user and not be writable by others')
        raw = os.read(descriptor, _MAX_BYTES + 1)
    finally:
        os.close(descriptor)
    return _parse(target, raw)


def _parse(target: Path, raw: bytes) -> Credential:
    """The credential in the bytes ``raw`` read from ``target``, or why it is not one."""
    if len(raw) > _MAX_BYTES:
        return Credential(problem=f'{target} is too large to be a credentials file')
    try:
        document = json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, ValueError):
        return Credential(problem=f'{target} is not valid JSON')
    if not isinstance(document, dict) or 'version' in document or 'origin' in document:
        return Credential(problem=f'{target} {_FORMAT_MESSAGE}')
    token, url = document.get('token'), document.get('url')
    if not isinstance(token, str) or not TOKEN_PATTERN.fullmatch(token):
        return Credential(problem=f'{target} holds no usable token')
    if not isinstance(url, str):
        return Credential(problem=f'{target} {_FORMAT_MESSAGE}')
    try:
        return Credential(url=validate_url(url), token=token, account=Account.from_json(document.get('account')))
    except UrlRefusedError as refused:
        return Credential(problem=f'the url in {target} cannot take a token: {refused}')
