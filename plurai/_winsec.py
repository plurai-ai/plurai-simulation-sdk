"""What the SDK asks Windows about the user, the profile and the access list of a file, and the rules it applies to the answers.

Two layers in one stdlib-only file, so a client that cannot import the SDK can copy it whole.

* The rules are plain functions over strings and numbers (:func:`profile_problem`, :func:`folder_problem`,
  :func:`access_problem`, :func:`blocked_sid`). They return a short code, or ``''`` when the answer is acceptable. The
  cases they must give are in ``tests/fixtures/credentials_windows_vectors.json``, which the other clients assert too.
* The calls (:func:`profile_directory`, :func:`drive_type`, :func:`user_sid`, :func:`file_security`, :func:`descriptor_security`,
  :func:`restrict`) are ``ctypes`` calls into ``shell32``, ``kernel32`` and ``advapi32``. They read the operating system,
  never the environment: ``HOME``, ``USERPROFILE`` and ``HOMEDRIVE`` are set by whoever launched the
  process, and a command a model runs sets its own.

Security is read and set on a *handle* opened without following links (``FILE_FLAG_OPEN_REPARSE_POINT``) and checked not
to be a reparse point, never on a path: a path can be swapped for a junction between the check and the call, and a call by
path then follows the junction. A call that fails raises :class:`WindowsSecurityError`; so does any call off Windows.
"""

from __future__ import annotations

import ctypes
import functools
import ntpath
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

SYSTEM_SID: Final = 'S-1-5-18'
ADMINISTRATORS_SID: Final = 'S-1-5-32-544'
DRIVE_FIXED: Final = 3

#: Group and service SIDs that are never the user running a process: a grant to one would open the file to many accounts.
_GROUP_SIDS: Final = frozenset({'S-1-1-0', 'S-1-5-11', 'S-1-5-7', 'S-1-5-2', 'S-1-5-4', 'S-1-5-13', 'S-1-5-15', 'S-1-2-0'})


class WindowsSecurityError(OSError):
    """A Windows call failed or is not available here. ``str()`` says which and never holds file content."""


class ReparsePointError(WindowsSecurityError):
    """The path is a symlink, a junction or another reparse point; it was not followed and nothing was changed."""


class OwnerError(WindowsSecurityError):
    """The path is owned by an account other than the ones the caller allowed; nothing was changed."""


@dataclass(frozen=True)
class FileSecurity:
    """The owner of a file or folder and the SIDs its access list allows. ``allowed`` holds ``'?'`` for an entry of a kind this module cannot read."""

    owner: str
    allowed: tuple[str, ...]
    dacl_present: bool


# --- the rules --------------------------------------------------------------------------------------------------


def blocked_sid(sid: str) -> bool:
    """Whether ``sid`` is a well-known group (Everyone, Authenticated Users, the ``BUILTIN`` groups ...) and so cannot be the user."""
    return sid in _GROUP_SIDS or sid.startswith('S-1-5-32-')


def _normal(path: str) -> str:
    text = path[4:] if path.startswith('\\\\?\\') and not path.startswith('\\\\?\\UNC\\') else path
    return ntpath.normcase(ntpath.normpath(text))


_VOLUME_ROOT: Final = re.compile(r'\\\\\?\\Volume\{[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}\}\\', re.IGNORECASE)


def local_root(path: str) -> str:
    """The root of the local volume ``path`` is on, or ``''`` when it is on none.

    ``C:\\`` for an ``X:\\...`` path (a long-path prefix is ignored) and ``\\\\?\\Volume{GUID}\\`` for a volume-GUID path, which is how a
    profile container mounted without a drive letter is named. UNC and device paths have no local root.
    """
    found = _VOLUME_ROOT.match(path)
    if found:
        return found.group(0)
    text = path[4:] if path.startswith('\\\\?\\') and not path.startswith('\\\\?\\UNC\\') else path
    if len(text) >= 3 and text[0].isascii() and text[0].isalpha() and text[1] == ':' and text[2] in '\\/':
        return text[:2] + '\\'
    return ''


def profile_problem(profile: str, drive_type: int, real_profile: str, real_drive_type: int) -> str:
    """Why ``profile`` (the operating system's answer for the user's profile folder) is not a place for a secret, or ``''``.

    ``real_profile`` is the profile with every link resolved and ``real_drive_type`` the drive type of its drive. A profile
    that is itself a link (a profile container, a junction to another drive) is accepted when its target is a fixed local
    folder: the owner and the access list of what is read are checked anyway.

    Codes: ``unusable`` (empty, or a control or line-separator character), ``network`` (the profile or its target is a UNC
    or device path), ``not_drive`` (not ``X:\\...``) and ``not_fixed`` (the drive of the profile or of its target is not
    ``DRIVE_FIXED``: a network, removable or RAM drive).
    """
    if not profile or any(ord(char) < 0x20 or ord(char) in (0x7F, 0x85, 0x2028, 0x2029) for char in profile):
        return 'unusable'
    if profile.startswith(('\\\\', '//')):
        return 'network'
    if not local_root(profile) or ':' in profile[2:]:
        return 'not_drive'
    if drive_type != DRIVE_FIXED:
        return 'not_fixed'
    if _normal(real_profile) == _normal(profile):
        return ''
    if real_profile.startswith('\\\\?\\UNC\\') or (real_profile.startswith(('\\\\', '//')) and not real_profile.startswith('\\\\?\\')):
        return 'network'
    if not local_root(real_profile):
        return 'not_drive'
    return '' if real_drive_type == DRIVE_FIXED else 'not_fixed'


def folder_problem(real_profile: str, real_folder: str) -> str:
    """Why the credentials folder, once links are resolved, is not ``<real_profile>\\.plurai`` (``moved``), or ``''``."""
    return '' if _normal(real_folder) == _normal(real_profile) + '\\.plurai' else 'moved'


def access_problem(owner: str, allowed: Sequence[str], dacl_present: bool, user: str) -> str:
    """Why a file or folder with this owner and access list is not the user's alone, or ``''``.

    Codes: ``owner`` (owned by neither the user nor the Administrators group, which owns what an elevated process creates),
    ``no_acl`` (no access list at all: everyone has full access) and ``others`` (an allow entry for any SID but the user,
    SYSTEM and Administrators, or one that cannot be read).
    """
    if owner not in (user, ADMINISTRATORS_SID):
        return 'owner'
    if not dacl_present:
        return 'no_acl'
    if any(sid not in (user, SYSTEM_SID, ADMINISTRATORS_SID) for sid in allowed):
        return 'others'
    return ''


def restricted_problem(allowed: Sequence[str], dacl_present: bool, user: str) -> str:
    """After the access list was cut to ``user``: ``''`` when it allows exactly the user, else ``not_restricted``."""
    return '' if dacl_present and set(allowed) == {user} else 'not_restricted'


_MESSAGES: Final = {
    'unusable': 'is not a usable path',
    'network': 'is a network or device path, which is not used for a credential',
    'not_drive': 'is not an absolute path on a drive letter',
    'not_fixed': 'is not on a fixed local drive, so it is not used for a credential',
    'moved': 'resolves to another folder through a link, which is not followed',
    'owner': 'is not owned by your account',
    'no_acl': 'has no access list, so every account can use it',
    'others': 'lets other accounts use it (only you, SYSTEM and Administrators may have access)',
    'not_restricted': 'could not be limited to your account',
}


def message(code: str, path: object) -> str:
    """``code`` as a sentence that names ``path``."""
    return f'{path} {_MESSAGES.get(code, "cannot be used")}'


# --- the calls --------------------------------------------------------------------------------------------------


class _Guid(ctypes.Structure):
    _fields_ = (('Data1', ctypes.c_uint32), ('Data2', ctypes.c_uint16), ('Data3', ctypes.c_uint16), ('Data4', ctypes.c_ubyte * 8))


class _AclSize(ctypes.Structure):
    _fields_ = (('AceCount', ctypes.c_uint32), ('AclBytesInUse', ctypes.c_uint32), ('AclBytesFree', ctypes.c_uint32))


class _FileTime(ctypes.Structure):
    _fields_ = (('Low', ctypes.c_uint32), ('High', ctypes.c_uint32))


class _FileInformation(ctypes.Structure):
    _fields_ = (
        ('Attributes', ctypes.c_uint32),
        ('Created', _FileTime),
        ('Accessed', _FileTime),
        ('Written', _FileTime),
        ('VolumeSerial', ctypes.c_uint32),
        ('SizeHigh', ctypes.c_uint32),
        ('SizeLow', ctypes.c_uint32),
        ('Links', ctypes.c_uint32),
        ('IndexHigh', ctypes.c_uint32),
        ('IndexLow', ctypes.c_uint32),
    )


_FOLDERID_PROFILE: Final = _Guid(0x5E6C858F, 0x0E22, 0x4760, (ctypes.c_ubyte * 8)(0x9A, 0xFE, 0xEA, 0x33, 0x17, 0xB6, 0x71, 0x73))
_TOKEN_QUERY: Final = 0x0008
_TOKEN_USER: Final = 1
_SE_FILE_OBJECT: Final = 1
_OWNER_AND_DACL: Final = 0x1 | 0x4
_ACL_SIZE_INFORMATION: Final = 2
_READ_CONTROL: Final = 0x00020000
_WRITE_DAC: Final = 0x00040000
_SHARE_ALL: Final = 0x7
_OPEN_EXISTING: Final = 3
_OPEN_LINK_ITSELF: Final = 0x02000000 | 0x00200000  # FILE_FLAG_BACKUP_SEMANTICS (a folder can be opened) | FILE_FLAG_OPEN_REPARSE_POINT
_REPARSE_POINT_ATTRIBUTE: Final = 0x400
_DACL_ONLY: Final = 0x4
_PROTECTED_DACL: Final = 0x80000000
_SDDL_REVISION: Final = 1
_SID_TEXT: Final = re.compile(r'S-1-\d+(?:-\d+)+')


@functools.cache
def _api() -> Any:  # noqa: ANN401 - a namespace of loaded libraries
    if sys.platform != 'win32':
        raise WindowsSecurityError('the Windows security calls are not available on this platform')
    from ctypes import POINTER, c_int, c_long, c_uint, c_void_p, c_wchar_p  # noqa: PLC0415

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    advapi32 = ctypes.WinDLL('advapi32', use_last_error=True)
    shell32 = ctypes.WinDLL('shell32', use_last_error=True)
    ole32 = ctypes.WinDLL('ole32', use_last_error=True)
    dword = ctypes.c_uint32
    kernel32.GetCurrentProcess.restype = c_void_p
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.CloseHandle.argtypes = [c_void_p]
    kernel32.CreateFileW.restype = c_void_p
    kernel32.CreateFileW.argtypes = [c_wchar_p, dword, dword, c_void_p, dword, dword, c_void_p]
    kernel32.GetFileInformationByHandle.argtypes = [c_void_p, POINTER(_FileInformation)]
    kernel32.LocalFree.restype = c_void_p
    kernel32.LocalFree.argtypes = [c_void_p]
    kernel32.GetDriveTypeW.restype = c_uint
    kernel32.GetDriveTypeW.argtypes = [c_wchar_p]
    advapi32.OpenProcessToken.argtypes = [c_void_p, dword, POINTER(c_void_p)]
    advapi32.GetTokenInformation.argtypes = [c_void_p, c_int, c_void_p, dword, POINTER(dword)]
    advapi32.ConvertSidToStringSidW.argtypes = [c_void_p, POINTER(c_wchar_p)]
    advapi32.GetSecurityInfo.restype = dword
    advapi32.GetSecurityInfo.argtypes = [c_void_p, c_int, dword, POINTER(c_void_p), POINTER(c_void_p), POINTER(c_void_p), POINTER(c_void_p), POINTER(c_void_p)]
    advapi32.SetSecurityInfo.restype = dword
    advapi32.SetSecurityInfo.argtypes = [c_void_p, c_int, dword, c_void_p, c_void_p, c_void_p, c_void_p]
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [c_wchar_p, dword, POINTER(c_void_p), POINTER(dword)]
    advapi32.GetSecurityDescriptorDacl.argtypes = [c_void_p, POINTER(ctypes.c_int), POINTER(c_void_p), POINTER(ctypes.c_int)]
    advapi32.GetAclInformation.argtypes = [c_void_p, c_void_p, dword, c_int]
    advapi32.GetAce.argtypes = [c_void_p, dword, POINTER(c_void_p)]
    shell32.SHGetKnownFolderPath.restype = c_long
    shell32.SHGetKnownFolderPath.argtypes = [POINTER(_Guid), dword, c_void_p, POINTER(c_wchar_p)]
    ole32.CoTaskMemFree.argtypes = [c_void_p]

    class Api:
        pass

    api = Api()
    api.kernel32, api.advapi32, api.shell32, api.ole32 = kernel32, advapi32, shell32, ole32
    return api


def _fail(what: str) -> WindowsSecurityError:
    return WindowsSecurityError(f'{what} failed (Windows error {ctypes.get_last_error()})')


def profile_directory() -> str:
    """The user's profile folder as the operating system names it (``FOLDERID_Profile``), not from any environment variable."""
    api = _api()
    out = ctypes.c_wchar_p()
    result = api.shell32.SHGetKnownFolderPath(ctypes.byref(_FOLDERID_PROFILE), 0, None, ctypes.byref(out))
    try:
        if result != 0 or not out.value:
            raise WindowsSecurityError(f'the profile folder cannot be read (HRESULT {result & 0xFFFFFFFF:#x})')
        return str(out.value)
    finally:
        api.ole32.CoTaskMemFree(ctypes.cast(out, ctypes.c_void_p))


def drive_type(root: str) -> int:
    """``GetDriveTypeW`` of ``root`` (``C:\\``): 3 is a fixed local drive, 4 a network drive."""
    return int(_api().kernel32.GetDriveTypeW(root))


def _sid_text(sid: int | None) -> str:
    api = _api()
    out = ctypes.c_wchar_p()
    if not sid or not api.advapi32.ConvertSidToStringSidW(sid, ctypes.byref(out)):
        raise _fail('ConvertSidToStringSidW')
    try:
        return str(out.value)
    finally:
        api.kernel32.LocalFree(ctypes.cast(out, ctypes.c_void_p))


def _token_sid(information_class: int) -> str:
    api = _api()
    token = ctypes.c_void_p()
    if not api.advapi32.OpenProcessToken(api.kernel32.GetCurrentProcess(), _TOKEN_QUERY, ctypes.byref(token)):
        raise _fail('OpenProcessToken')
    try:
        needed = ctypes.c_uint32()
        api.advapi32.GetTokenInformation(token, information_class, None, 0, ctypes.byref(needed))
        if needed.value == 0:
            raise _fail('GetTokenInformation')
        buffer = ctypes.create_string_buffer(needed.value)
        if not api.advapi32.GetTokenInformation(token, information_class, buffer, needed, ctypes.byref(needed)):
            raise _fail('GetTokenInformation')
        return _sid_text(ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0])
    finally:
        api.kernel32.CloseHandle(token)


def user_sid() -> str:
    """The SID of the user running the process, from the process token (never from an environment variable)."""
    user = _token_sid(_TOKEN_USER)
    if blocked_sid(user):
        raise WindowsSecurityError('the process token names a group, not a user')
    return user


def _open_link_free(path: object, access: int) -> int:
    """A handle on ``path`` itself (a link is opened, not followed), checked not to be a reparse point; the caller closes it.

    Raises :class:`ReparsePointError` for a symlink, junction or other reparse point.
    """
    api = _api()
    handle = api.kernel32.CreateFileW(str(path), access, _SHARE_ALL, None, _OPEN_EXISTING, _OPEN_LINK_ITSELF, None)
    if handle is None or handle in (0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF):
        raise _fail('CreateFileW')
    try:
        info = _FileInformation()
        if not api.kernel32.GetFileInformationByHandle(handle, ctypes.byref(info)):
            raise _fail('GetFileInformationByHandle')
        if info.Attributes & _REPARSE_POINT_ATTRIBUTE:
            raise ReparsePointError('the path is a symlink or junction')
    except BaseException:
        api.kernel32.CloseHandle(handle)
        raise
    return int(handle)


def _security_of(handle: int) -> FileSecurity:
    """The owner and the allowed SIDs of the object ``handle`` is open on."""
    api = _api()
    owner, dacl, descriptor = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    status = api.advapi32.GetSecurityInfo(
        handle, _SE_FILE_OBJECT, _OWNER_AND_DACL, ctypes.byref(owner), None, ctypes.byref(dacl), None, ctypes.byref(descriptor)
    )
    if status != 0:
        raise WindowsSecurityError(f'GetSecurityInfo failed (Windows error {status})')
    try:
        owner_sid = _sid_text(owner.value)
        if not dacl.value:
            return FileSecurity(owner_sid, (), dacl_present=False)
        size = _AclSize()
        if not api.advapi32.GetAclInformation(dacl, ctypes.byref(size), ctypes.sizeof(size), _ACL_SIZE_INFORMATION):
            raise _fail('GetAclInformation')
        allowed: list[str] = []
        for index in range(size.AceCount):
            ace = ctypes.c_void_p()
            if not api.advapi32.GetAce(dacl, index, ctypes.byref(ace)):
                raise _fail('GetAce')
            kind = ctypes.c_ubyte.from_address(ace.value).value  # type: ignore[arg-type]
            if kind in (1, 10):  # a deny entry restricts, it grants nothing
                continue
            # An allow entry (plain 0, callback 9) holds its SID after the 4-byte header and the 4-byte mask; any other kind is not read.
            allowed.append(_sid_text(ace.value + 8) if kind in (0, 9) and ace.value else '?')
        return FileSecurity(owner_sid, tuple(allowed), dacl_present=True)
    finally:
        api.kernel32.LocalFree(descriptor)


def descriptor_security(descriptor: int) -> FileSecurity:
    """The owner and the allowed SIDs of the object an open file ``descriptor`` is on, read on the descriptor's own handle.

    No path is involved, so the object checked is the object that will be read.
    """
    if sys.platform != 'win32':
        raise WindowsSecurityError('the Windows security calls are not available on this platform')
    import msvcrt  # noqa: PLC0415 - Windows only

    return _security_of(msvcrt.get_osfhandle(descriptor))  # type: ignore[attr-defined]


def file_security(path: object) -> FileSecurity:
    """The owner of ``path`` and the SIDs its access list allows, read on a handle that is not a reparse point.

    Raises :class:`ReparsePointError` when ``path`` is a link.
    """
    handle = _open_link_free(path, _READ_CONTROL)
    try:
        return _security_of(handle)
    finally:
        _api().kernel32.CloseHandle(handle)


def restrict(path: object, user: str, *, directory: bool, owners: Sequence[str] | None = None) -> FileSecurity:
    """Make ``path`` accessible to ``user`` alone, protected from inheritance, and return the list as read back.

    Everything happens on one handle opened without following links, so a path swapped for a junction after the caller
    looked at it changes nothing: :class:`ReparsePointError` is raised and no list is touched. With ``owners``, an object
    owned by any other account raises :class:`OwnerError` before its list is changed.
    """
    if not _SID_TEXT.fullmatch(user) or blocked_sid(user):
        raise WindowsSecurityError('the user SID is not usable')
    api = _api()
    handle = _open_link_free(path, _READ_CONTROL | _WRITE_DAC)
    descriptor, dacl, present, defaulted = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_int(), ctypes.c_int()
    try:
        if owners is not None and _security_of(handle).owner not in owners:
            raise OwnerError('the path is owned by another account')
        sddl = f'D:P(A;OICI;FA;;;{user})' if directory else f'D:P(A;;FA;;;{user})'
        if not api.advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, _SDDL_REVISION, ctypes.byref(descriptor), None):
            raise _fail('ConvertStringSecurityDescriptorToSecurityDescriptorW')
        if not api.advapi32.GetSecurityDescriptorDacl(descriptor, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted)) or not present.value:
            raise _fail('GetSecurityDescriptorDacl')
        status = api.advapi32.SetSecurityInfo(handle, _SE_FILE_OBJECT, _DACL_ONLY | _PROTECTED_DACL, None, None, dacl, None)
        if status != 0:
            raise WindowsSecurityError(f'SetSecurityInfo failed (Windows error {status})')
        return _security_of(handle)
    finally:
        if descriptor.value:
            api.kernel32.LocalFree(descriptor)
        api.kernel32.CloseHandle(handle)
