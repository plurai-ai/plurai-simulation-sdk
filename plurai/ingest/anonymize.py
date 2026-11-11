"""Anonymization of extractor-output JSON — the scrub applied under an env's **anonymized**
posture.

This is the **egress chokepoint** for an env that opted in: the local scrub every byte of
customer data passes through before it can leave the machine. Whether it runs is governed by
the env's anonymization posture (:mod:`plurai.ingest.posture`); *within* an anonymized posture
it is fail-closed, with no raw path — the full :class:`Anonymizer` owns the file/parallel egress
methods. A raw-posture env never builds an :class:`Anonymizer` at all (see :func:`plurai.ingest.cli.run_ingest`).

Two steps, one keyed replacement:

* **Detect** — PII (person names, emails, phone numbers, SSNs, locations, …) is detected
  with Presidio's default ``AnalyzerEngine`` (spaCy NER + its built-in regex
  recognizers). There is no declared-schema / identifier-format layer.
* **Replace** — each detected span is replaced with a **one-way keyed hash**
  (:func:`keyed_hash`) seeded by a fixed per-workspace **pseudonym key**
  (``.plurai/anon.key``, minted on first run, gitignored, never uploaded). Because the
  hash is a **pure, stateless** function of ``value + key``, the same real value maps to
  the same pseudonym across files and across separate runs (within-trace / closed-loop
  consistency) with **no shared in-memory map**, so anonymization
  :func:`parallelizes <anonymize_files>` cleanly.

Ingested traces/RAG are used only to *learn* on — nothing downstream resolves a pseudonym
against a live/mock system and nothing joins the same entity across records — so opaque
ids (e.g. ``ACC_12345``, ``user_654``) are **not** treated as PII: they pass through
unless a built-in recognizer flags them. The PII that matters (names, emails, phones,
SSNs, addresses) is caught by Presidio either way.

The pipeline is **fail-closed**: if Presidio or a spaCy model is not installed it raises
(:class:`AnonymizationDependencyError`) rather than proceed.

**Binary assets** (image/PDF attachments that are not scrubbable UTF-8 text) are handled
best-effort by :meth:`Anonymizer._scrub_binary`: PNG/JPEG are EXIF/metadata-stripped
AND OCR-redacted, PDFs are rasterized-redacted-and-re-assembled image-only (discarding the
text layer/XMP). Best-effort is **NOT** pass-through — an unsupported/corrupt asset or a missing
optional dep fails closed (never egresses raw bytes). The loud residuals: OCR misses
stylized/rotated/low-contrast/non-Latin text; visual PII — faces/signatures/logos — is out of
scope; redaction is destructive, not a reversible pseudonym.

The ``sha256(key)`` **fingerprint** (:func:`key_fingerprint`) is written to each
artifact's audit record so VERIFY can detect a rotated/lost key
(:func:`verify_key_fingerprint`) instead of silently reading as "different entity".
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import io
import json
import logging
import os
import secrets
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from plurai import _private_files
from plurai._platform import is_windows

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

#: Directory (relative to the workspace root) that holds ingestion runtime state.
KEY_DIR_NAME = '.plurai'
#: The per-workspace pseudonym-key file. Gitignored; never uploaded.
KEY_FILENAME = 'anon.key'
#: Pseudonym key length in bytes (256-bit).
KEY_BYTES = 32
#: Truncated length of the hex key fingerprint written to the audit log.
FINGERPRINT_LEN = 16
#: Length of the free-text replacement token (hex chars of the keyed hash).
FREE_TEXT_TOKEN_LEN = 16

_MISSING_PRESIDIO_MSG = (
    'Anonymization requires Presidio, which is not installed. Ingestion is '
    'fail-closed and refuses to upload un-anonymized customer data. Install the '
    "optional extra: `pip install 'plurai-simulation-sdk[anonymize]'`, then download a spaCy "
    'English model, e.g. `python -m spacy download en_core_web_lg`.'
)
_MISSING_MODEL_MSG = (
    'Anonymization requires a spaCy English model, which is not installed. Ingestion '
    'is fail-closed. Install one, e.g. `python -m spacy download en_core_web_lg`.'
)
_MISSING_IMAGE_REDACTOR_MSG = (
    'Binary-asset anonymization (image OCR-redaction / PDF rasterization) requires the '
    'binary extra AND two system binaries, none of which are present. Ingestion is '
    'fail-closed and refuses to upload un-scrubbed bytes. Install the optional extra: '
    "`pip install 'plurai-simulation-sdk[anonymize-binary]'`, then install the Tesseract OCR engine and "
    'Poppler (`pdftoppm`) — Linux: `apt-get install tesseract-ocr poppler-utils`; '
    'macOS: `brew install tesseract poppler`.'
)


# --------------------------------------------------------------------------- #
# Exceptions                                                                   #
# --------------------------------------------------------------------------- #


class AnonymizationDependencyError(ImportError):
    """Raised (fail-closed) when Presidio or a spaCy model is not installed."""


class KeyFingerprintMismatchError(ValueError):
    """Raised when the active pseudonym key does not match an artifact's fingerprint."""


class DocExampleBinaryError(ValueError):
    """Raised (fail-closed) when binary content cannot be safely scrubbed.

    Supported binary assets (PNG/JPEG/PDF) are best-effort scrubbed in place — OCR-redacted
    and metadata-stripped, NOT refused. This is raised only when scrubbing is not
    possible: invalid base64, an unrecognized/unsupported/corrupt magic, a supported asset the
    loader cannot decode, or a page-less PDF — anonymization refuses rather than egress
    un-scrubbed bytes. (A missing binary *dependency* raises :class:`AnonymizationDependencyError`.)
    """


# --------------------------------------------------------------------------- #
# Pure keyed-hash core (no Presidio; the load-bearing consistency mechanism)   #
# --------------------------------------------------------------------------- #


def keyed_hash(value: str, key: bytes) -> str:
    """One-way keyed pseudonym for PII: ``HMAC-SHA256(key, value)`` hex.

    Pure and stateless: same ``value`` + ``key`` always yields the same digest, so
    consistency holds across files and parallel workers with no shared map.
    """
    return hmac.new(key, value.encode('utf-8'), hashlib.sha256).hexdigest()


# --------------------------------------------------------------------------- #
# Pseudonym key management + fingerprint                                        #
# --------------------------------------------------------------------------- #

# Makes ``.plurai/`` self-ignoring so the pseudonym key + audit log can never be
# committed: ``*`` ignores everything under it, ``!.gitignore`` keeps this file itself
# trackable (so the intent is visible/committable). Single source of truth — callers
# and skills don't each have to remember to gitignore the secret.
_KEY_DIR_GITIGNORE = '# Auto-generated by plurai ingestion — never commit ingestion secrets/artifacts.\n*\n!.gitignore\n'


def _ensure_key_dir_gitignored(key_dir: Path) -> None:
    """Write ``.plurai/.gitignore`` (self-ignoring) if absent. Best-effort."""
    gitignore = key_dir / '.gitignore'
    if gitignore.exists():
        return
    try:
        gitignore.write_text(_KEY_DIR_GITIGNORE, encoding='utf-8')
    except OSError as e:  # pragma: no cover - best effort on exotic filesystems
        # This guards a SECRET (the pseudonym key) from being committed — warn loudly
        # rather than silently leave the key directory trackable.
        logger.warning('Could not write %s (%s); ensure %s is git-ignored — it holds the pseudonym key.', gitignore, e, key_dir)


def load_or_create_key(workspace_root: str | Path = '.') -> bytes:
    """Return the workspace pseudonym key, minting it on first run.

    The key lives at ``<workspace_root>/.plurai/anon.key`` (hex-encoded). It is
    created with ``0o600`` permissions so the secret is not world/group readable, and
    ``.plurai/`` is made self-ignoring (a ``.plurai/.gitignore``) so the key + audit
    log are never committed — only the fingerprint travels.
    """
    key_dir = Path(workspace_root) / KEY_DIR_NAME
    key_file = key_dir / KEY_FILENAME

    if key_file.exists():
        if is_windows():  # an existing key and folder are checked and limited again, not trusted: a failure stops the read
            _private_files.restrict_directory(key_dir)
            _private_files.restrict_to_user(key_file, require_owner=True)
        _ensure_key_dir_gitignored(key_dir)  # an existing key file may have no .gitignore beside it
        text = key_file.read_text(encoding='utf-8').strip()
        try:
            key = bytes.fromhex(text)
        except ValueError as e:
            raise ValueError(f'{key_file} is not a valid hex pseudonym key. Remove it to mint a fresh one (this invalidates prior artifacts).') from e
        if len(key) != KEY_BYTES:
            raise ValueError(f'{key_file} holds a {len(key)}-byte key; expected {KEY_BYTES}. Remove it to mint a fresh one.')
        return key

    key_dir.mkdir(parents=True, exist_ok=True)
    try:
        key_dir.chmod(0o700)
    except OSError:  # pragma: no cover - best effort on exotic filesystems
        pass
    if is_windows():  # no modes there: the folder's access list is cut to the user, and the key inherits it
        _private_files.restrict_directory(key_dir)
    _ensure_key_dir_gitignored(key_dir)  # ignore the dir BEFORE the secret is written into it

    hex_key = secrets.token_hex(KEY_BYTES)
    try:
        fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        # Lost a mint race with a concurrent first run — read the winner's key.
        return load_or_create_key(workspace_root)
    try:
        os.write(fd, hex_key.encode('ascii'))
    finally:
        os.close(fd)
    key_file.chmod(0o600)
    return bytes.fromhex(hex_key)


def key_fingerprint(key: bytes) -> str:
    """Public, non-reversible fingerprint of the pseudonym key for the audit log."""
    return hashlib.sha256(key).hexdigest()[:FINGERPRINT_LEN]


def verify_key_fingerprint(key: bytes, expected: str) -> None:
    """Fail-closed check that ``key`` produced an artifact's recorded fingerprint.

    Raises :class:`KeyFingerprintMismatchError` on mismatch so a rotated/lost key
    surfaces as an error rather than silently reading as a "different entity".
    """
    actual = key_fingerprint(key)
    if not hmac.compare_digest(actual, expected):
        raise KeyFingerprintMismatchError(
            f'Pseudonym key fingerprint {actual!r} does not match the artifact fingerprint {expected!r}. '
            'The .plurai/anon.key was rotated or lost; prior artifacts must be re-ingested with the current key.'
        )


# --------------------------------------------------------------------------- #
# Presidio integration (fail-closed)                                           #
# --------------------------------------------------------------------------- #


def _import_presidio() -> SimpleNamespace:
    """Import the Presidio surface we need, or raise the fail-closed error.

    The single dependency guard for the whole module: it raises
    :class:`AnonymizationDependencyError` (with the actionable install message) on a
    missing import, so every caller can just call it and let the error propagate.
    """
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig
    except ImportError as e:
        raise AnonymizationDependencyError(_MISSING_PRESIDIO_MSG) from e
    return SimpleNamespace(
        AnalyzerEngine=AnalyzerEngine,
        NlpEngineProvider=NlpEngineProvider,
        AnonymizerEngine=AnonymizerEngine,
        OperatorConfig=OperatorConfig,
    )


def _available_spacy_model() -> str | None:
    """Return the best installed spaCy English model name, or None if none present."""
    try:
        import spacy
    except ImportError:
        return None
    for name in ('en_core_web_lg', 'en_core_web_md', 'en_core_web_sm'):
        if spacy.util.is_package(name):
            return name
    return None


def build_analyzer(*, model: str | None = None) -> Any:  # noqa: ANN401 - presidio AnalyzerEngine (optional dep)
    """Build a fail-closed Presidio ``AnalyzerEngine`` (built-in recognizers only).

    Uses the spaCy NER engine plus Presidio's default built-in recognizers (person,
    email, phone, SSN, credit-card, …). Raises :class:`AnonymizationDependencyError`
    if Presidio or a spaCy model is missing.
    """
    presidio = _import_presidio()

    model_name = model or _available_spacy_model()
    if model_name is None:
        raise AnonymizationDependencyError(_MISSING_MODEL_MSG)

    provider = presidio.NlpEngineProvider(nlp_configuration={'nlp_engine_name': 'spacy', 'models': [{'lang_code': 'en', 'model_name': model_name}]})
    nlp_engine = provider.create_engine()
    return presidio.AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=['en'])


def build_anonymizer() -> Any:  # noqa: ANN401 - presidio AnonymizerEngine (optional dep)
    """Build a fail-closed Presidio ``AnonymizerEngine``."""
    return _import_presidio().AnonymizerEngine()


# --------------------------------------------------------------------------- #
# Binary-asset anonymization (image OCR-redaction / PDF rasterize)             #
# --------------------------------------------------------------------------- #

#: The v1 supported binary formats, keyed by their leading magic bytes. Magic bytes are
#: authoritative (a caller-supplied ``media_hint`` is only a fallback label) so a
#: mislabeled/renamed asset can neither smuggle an unsupported type past the scrub nor
#: be mis-dispatched. Anything else fails closed (:class:`DocExampleBinaryError`).
_MEDIA_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b'\x89PNG\r\n\x1a\n', 'png'),
    (b'\xff\xd8\xff', 'jpeg'),
    (b'%PDF-', 'pdf'),
)


def _sniff_media(raw: bytes) -> str | None:
    """Detect a supported binary format from leading magic bytes, else ``None``.

    ``None`` (unrecognized/unsupported magic) is the fail-closed signal — v1 supports
    only PNG, JPEG and PDF; every other type refuses rather than egress un-scrubbed.
    ``imghdr`` is deliberately NOT used (removed in Python 3.13).
    """
    for magic, media in _MEDIA_MAGIC:
        if raw.startswith(magic):
            return media
    return None


def _import_image_redactor() -> SimpleNamespace:
    """Import the binary-anonymization surface we need, or raise the fail-closed error.

    Mirrors :func:`_import_presidio`: the single dependency guard for the binary path.
    Raises :class:`AnonymizationDependencyError` (with the actionable install message)
    when the ``anonymize-binary`` extra is not installed, so callers just call it and
    let the error propagate. The two grouped exception tuples let :meth:`Anonymizer._scrub_binary`
    tell a MISSING system binary (Tesseract/Poppler absent at CALL time → dependency error)
    apart from a CORRUPT/undecodable asset (→ fail-closed :class:`DocExampleBinaryError`),
    which matters because ``TesseractNotFoundError`` is itself an ``OSError`` subclass and
    must be caught FIRST so it is never mislabeled as an unprocessable asset.
    """
    try:
        from PIL import Image, UnidentifiedImageError
        from pdf2image import convert_from_bytes
        from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError, PopplerNotInstalledError
        from presidio_image_redactor import ImageAnalyzerEngine, ImageRedactorEngine
        from pytesseract import TesseractNotFoundError
    except ImportError as e:
        raise AnonymizationDependencyError(_MISSING_IMAGE_REDACTOR_MSG) from e
    return SimpleNamespace(
        Image=Image,
        convert_from_bytes=convert_from_bytes,
        ImageAnalyzerEngine=ImageAnalyzerEngine,
        ImageRedactorEngine=ImageRedactorEngine,
        #: System binary absent at call time (OCR engine / Poppler) → AnonymizationDependencyError.
        #: PopplerNotInstalledError is the base of PDFInfoNotInstalledError, so it covers both.
        missing_binary_errors=(TesseractNotFoundError, PopplerNotInstalledError),
        #: Supported magic but the loader/redactor could not open/decode it → DocExampleBinaryError.
        #: OSError/ValueError cover truncated images (PIL raises OSError on .load()) and stray
        #: decoder value errors; the missing-binary tuple above is matched FIRST.
        unprocessable_errors=(UnidentifiedImageError, PDFPageCountError, PDFSyntaxError, OSError, ValueError),
    )


def _strip_image_metadata(img: Any) -> bytes:  # noqa: ANN401 - PIL.Image.Image (optional dep)
    """Re-encode a PIL image dropping ALL metadata → bytes.

    Saving WITHOUT the ``exif=``/``pnginfo=`` kwargs makes Pillow write neither, so EXIF
    (including GPS + any embedded IFD1 thumbnail), XMP, and PNG text chunks are all
    discarded. The format is preserved from ``img.format`` (defaulting to PNG), so a PNG
    stays a PNG and a JPEG stays a JPEG.
    """
    fmt = (img.format or 'PNG').upper()
    if fmt == 'JPG':
        fmt = 'JPEG'
    buf = io.BytesIO()
    img.save(buf, format=fmt)  # no exif=/pnginfo= → all metadata dropped
    return buf.getvalue()


def _scrub_image(raw: bytes, redactor: Any, ir: SimpleNamespace, media: str) -> bytes:  # noqa: ANN401 - presidio ImageRedactorEngine (optional dep)
    """OCR-redact PII regions in an image AND strip its metadata → bytes.

    Opens the image, redacts detected PII (black fill) via the shared analyzer, then
    re-encodes without metadata. Converting to RGB normalizes palette/grayscale/alpha
    modes so the fill and re-encode are well-defined for every input.
    """
    with ir.Image.open(io.BytesIO(raw)) as opened:
        opened.load()  # force decode now so a truncated image raises here (fail-closed), not later
        redacted = redactor.redact(opened.convert('RGB'), fill=(0, 0, 0), language='en')
    redacted.format = 'PNG' if media == 'png' else 'JPEG'
    return _strip_image_metadata(redacted)


def _scrub_pdf(raw: bytes, redactor: Any, ir: SimpleNamespace) -> bytes:  # noqa: ANN401 - presidio ImageRedactorEngine (optional dep)
    """Rasterize each PDF page, OCR-redact it, and re-assemble an IMAGE-ONLY PDF → bytes.

    Rasterizing (dpi=250) then re-assembling from images DELIBERATELY discards the text
    layer, annotations, form fields and XMP — the very places metadata PII text-extraction
    would leak — leaving only redacted page rasters. Page count is preserved.
    """
    pages = ir.convert_from_bytes(raw, dpi=250)
    if not pages:  # a valid-magic PDF that yields no pages is unusable → fail closed
        raise DocExampleBinaryError('PDF asset yielded no pages; refusing to egress un-scrubbed bytes (fail-closed).')
    redacted = [redactor.redact(page.convert('RGB'), fill=(0, 0, 0), language='en') for page in pages]
    buf = io.BytesIO()
    # Image-only PDF, carrying no doc-info/XMP beyond Pillow's creation timestamps.
    redacted[0].save(buf, format='PDF', save_all=True, append_images=redacted[1:])
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Structure-preserving JSON walk (shared by the full scrubber)                  #
# --------------------------------------------------------------------------- #


class TreeHooks(Protocol):
    """Follows a JSON walk position by position, for a caller that needs more than the scrubbed tree.

    ``where`` is whatever the hooks returned for the position above; the root's is ``None``. ``descend`` names
    the position of ``parent[key]``, with ``key_kept`` false when the scrub renamed the key. ``leaf`` decides a
    string leaf: it returns the text to write, normally ``scrub_value(value)``, and may return ``value`` itself
    to let a leaf ride through.
    """

    def descend(self, parent: dict[str, Any], key: str, key_kept: bool, where: Any) -> Any: ...  # noqa: ANN401

    def leaf(self, where: Any, value: str, scrub_value: Callable[[str], str]) -> str: ...  # noqa: ANN401


def _scrub_tree(
    obj: Any,  # noqa: ANN401 - arbitrary decoded JSON
    scrub_value: Callable[[str], str],
    scrub_key: Callable[[str], str],
    hooks: TreeHooks | None = None,
    where: Any = None,  # noqa: ANN401 - the hooks' own position type
) -> Any:  # noqa: ANN401
    """Recursively scrub decoded JSON, scrubbing both string leaves AND string keys.

    Containers are mutated in place and returned as the same object (halving peak
    memory); a dict is rebuilt only when a key actually changes (deterministic
    last-writer-wins on the rare pseudonym collision). ``hooks`` observe the walk and may exempt a leaf; they
    never change which keys are scrubbed, which ``scrub_key`` alone decides.
    """
    if isinstance(obj, dict):
        items: list[tuple[Any, Any]] = []
        keys_changed = False
        for k, v in obj.items():
            new_k = scrub_key(k) if isinstance(k, str) and k else k
            inner = hooks.descend(obj, k, new_k == k, where) if hooks is not None and isinstance(k, str) else where
            items.append((new_k, _scrub_tree(v, scrub_value, scrub_key, hooks, inner)))
            if new_k != k:
                keys_changed = True
        if not keys_changed:
            for new_k, new_v in items:
                obj[new_k] = new_v
            return obj
        return dict(items)  # rebuilt; last-writer-wins on the rare pseudonym collision
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = _scrub_tree(v, scrub_value, scrub_key, hooks, where)
        return obj
    if isinstance(obj, str):
        return scrub_value(obj) if hooks is None else hooks.leaf(where, obj, scrub_value)
    return obj


# --------------------------------------------------------------------------- #
# Full anonymizer (Presidio detect → keyed-hash replace) — owns the egress path #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, kw_only=True)
class AnonymizerSpec:
    """Picklable recipe for building a full :class:`Anonymizer`, in-process or in a worker.

    Presidio engines are not picklable, so the parallel path ships this spec to each
    worker and rebuilds the engines ONCE per worker (see :func:`anonymize_files`).
    :meth:`Anonymizer.from_spec` is the single place the key + engines are derived, so
    the in-process and parallel paths never drift.
    """

    workspace_root: str = '.'
    model: str | None = None


class Anonymizer:
    """Full PII scrub — Presidio detect → keyed-hash replace — and the egress path.

    It ALWAYS runs the Presidio engines (required), so its file/JSON methods are a
    complete scrub. Build it via :meth:`create` / :meth:`from_spec` (fail-closed on
    missing deps).
    """

    def __init__(
        self,
        key: bytes,
        analyzer: Any,  # noqa: ANN401 - presidio AnalyzerEngine (optional dep)
        anonymizer: Any,  # noqa: ANN401 - presidio AnonymizerEngine (optional dep)
    ) -> None:
        self._key = key
        self._analyzer = analyzer
        self._anonymizer = anonymizer
        self._kh_cache: dict[str, str] = {}
        self._key_cache: dict[str, str] = {}
        self._operators = self._build_operators()
        # Built lazily on the first binary asset (the base text path must not require the
        # optional binary extra) and cached like ``_operators``; reuses ``self._analyzer``.
        self._image_redactor: Any = None

    @classmethod
    def from_spec(cls, spec: AnonymizerSpec) -> Anonymizer:
        """Build a full anonymizer from a picklable spec (the single derivation point)."""
        key = load_or_create_key(spec.workspace_root)
        analyzer = build_analyzer(model=spec.model)
        anonymizer = build_anonymizer()
        return cls(key, analyzer, anonymizer)

    @classmethod
    def create(cls, workspace_root: str | Path = '.', *, model: str | None = None) -> Anonymizer:
        """Build the full anonymizer for a workspace (fail-closed) — called only under anonymized posture."""
        return cls.from_spec(AnonymizerSpec(workspace_root=str(Path(workspace_root).resolve()), model=model))

    def _kh(self, value: str) -> str:
        """Memoized keyed-hash token (free-text PII)."""
        cached = self._kh_cache.get(value)
        if cached is None:
            cached = keyed_hash(value, self._key)[:FREE_TEXT_TOKEN_LEN]
            self._kh_cache[value] = cached
        return cached

    def pseudonym(self, value: str) -> str:
        """The keyed-hash token the scrub writes for ``value``, for a name that is not content.

        Archive member names are file names, and a file named after an id carries that id past a
        content-only scrub. Naming the member by this token keeps it stable across runs under one
        key. It is not counted in :attr:`scrub_summary`, which reports content only.
        """
        return keyed_hash(value, self._key)[:FREE_TEXT_TOKEN_LEN]

    def _build_operators(self) -> dict[str, Any]:
        presidio = _import_presidio()
        return {'DEFAULT': presidio.OperatorConfig('custom', {'lambda': self._kh})}

    @property
    def key_fingerprint(self) -> str:
        """The audit-log fingerprint of this anonymizer's pseudonym key."""
        return key_fingerprint(self._key)

    @property
    def scrub_summary(self) -> dict[str, int]:
        """Counts for the disclosure preview / audit — how much was pseudonymized, never the values.

        ``freetext_values`` is the distinct-value count so far, so read this AFTER scrubbing
        to summarize a run.
        """
        return {'freetext_values': len(self._kh_cache)}

    def anonymize_value(self, value: Any) -> Any:  # noqa: ANN401 - JSON leaf of any type
        """Anonymize a single JSON string leaf; non-strings pass through unchanged.

        Detects PII with Presidio (spaCy NER + built-in recognizers) and replaces each
        detected span with a keyed hash. Text with no detected PII passes through
        unchanged.
        """
        if not isinstance(value, str) or not value:
            return value
        results = self._analyzer.analyze(text=value, language='en')
        if not results:
            return value
        return self._anonymizer.anonymize(text=value, analyzer_results=results, operators=self._operators).text

    def _scrub_key(self, key: str) -> str:
        """Memoized key scrub (keys are low-cardinality field names, so this is cheap)."""
        cached = self._key_cache.get(key)
        if cached is None:
            cached = self.anonymize_value(key)
            self._key_cache[key] = cached
        return cached

    def anonymize_obj(
        self,
        obj: Any,  # noqa: ANN401 - arbitrary decoded JSON
        *,
        exempt_key: Callable[[str], bool] | None = None,
        hooks: TreeHooks | None = None,
    ) -> Any:  # noqa: ANN401
        """Anonymize decoded JSON (string leaves AND keys), preserving structure.

        ``exempt_key`` names dict keys that keep their spelling (a key an agent description already declares);
        a leaf value is never exempted by it. ``hooks`` follow the walk (see :class:`TreeHooks`).
        """
        scrub_key = self._scrub_key if exempt_key is None else (lambda k: k if exempt_key(k) else self._scrub_key(k))
        return _scrub_tree(obj, self.anonymize_value, scrub_key, hooks)

    def anonymize_file(self, path: str | Path) -> None:
        """Scrub a file in place (atomic), dispatching on content type.

        Real ingest inputs are mixed (a bundle carries ``record.json`` + markdown
        ``ROLES.md`` + binary attachments), so this is not JSON-only:

        * parses as JSON → structure-preserving tree scrub (keys + string leaves);
        * otherwise UTF-8 text (e.g. markdown) → free-text PII scrub over the whole body;
        * non-UTF-8 binary → best-effort binary-asset scrub (image OCR-redaction / PDF
          rasterization) written back atomically as BYTES; an unsupported/corrupt
          asset or a missing binary dep FAILS CLOSED (:class:`DocExampleBinaryError` /
          :class:`AnonymizationDependencyError`) and nothing is written (the scrub runs
          BEFORE any temp file is created, so the original is untouched on failure).
        """
        p = Path(path)
        raw = p.read_bytes()
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            scrubbed_bytes = self._scrub_binary(raw, media_hint=p.suffix)  # fail-closed; never returns raw
            tmp = p.with_name(f'{p.name}.tmp')
            tmp.write_bytes(scrubbed_bytes)
            tmp.replace(p)  # atomic in-place swap
            return
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            scrubbed = self.anonymize_value(text)  # free-text body (markdown, plain text, …)
        else:
            scrubbed = json.dumps(self.anonymize_obj(data), ensure_ascii=False)
        tmp = p.with_name(f'{p.name}.tmp')
        tmp.write_text(scrubbed, encoding='utf-8')
        tmp.replace(p)  # atomic in-place swap

    def anonymize_bytes(self, raw: bytes, *, media_hint: str | None = None) -> bytes:
        """Scrub one document's bytes as the document itself, never as structured data.

        UTF-8 text gets the free-text scrub over the whole body, even when it parses as JSON:
        the tree scrub of :meth:`anonymize_file` would re-serialize it, so the bytes would no
        longer be the document. Anything else gets the best-effort binary scrub, which fails
        closed (:class:`DocExampleBinaryError` / :class:`AnonymizationDependencyError`) rather
        than return un-scrubbed bytes. ``media_hint`` labels the asset in that error.
        """
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            return self._scrub_binary(raw, media_hint=media_hint)
        return self.anonymize_value(text).encode('utf-8')

    def _scrub_binary(self, raw: bytes, *, media_hint: str | None = None) -> bytes:
        """Best-effort scrub of a binary asset, returning scrubbed bytes.

        The ONE shared binary path every egress call site (:meth:`anonymize_file` and
        :meth:`anonymize_bytes`) delegates to. Dispatch is on MAGIC BYTES
        (``media_hint`` is only a fallback label for the error): PNG/JPEG are
        EXIF/metadata-stripped AND OCR-redacted; PDF is rasterized, redacted and
        re-assembled image-only. **Best-effort is NOT pass-through** — this NEVER returns
        raw/un-scrubbed bytes:

        * unrecognized/unsupported/corrupt magic, or supported magic the loader cannot
          open → :class:`DocExampleBinaryError` (fail closed);
        * the ``anonymize-binary`` extra, or the Tesseract/Poppler system binaries, absent
          → :class:`AnonymizationDependencyError`.

        The redactor reuses ``self._analyzer`` (built lazily + cached) so image and text
        PII detection stay in parity.
        """
        media = _sniff_media(raw)  # magic wins; media_hint is only a fallback label for the error
        if media is None:
            raise DocExampleBinaryError(
                f'asset (hint={media_hint!r}) is not a recognized/supported binary type (png/jpeg/pdf); refusing to egress un-scrubbed bytes (fail-closed).'
            )
        ir = _import_image_redactor()  # AnonymizationDependencyError if the binary extra is absent
        if self._image_redactor is None:
            image_analyzer = ir.ImageAnalyzerEngine(analyzer_engine=self._analyzer)  # SAME analyzer as text
            self._image_redactor = ir.ImageRedactorEngine(image_analyzer_engine=image_analyzer)
        try:
            if media == 'pdf':
                return _scrub_pdf(raw, self._image_redactor, ir)
            return _scrub_image(raw, self._image_redactor, ir, media)
        except DocExampleBinaryError:
            raise  # already the fail-closed error (e.g. a page-less PDF) — do not re-wrap
        except ir.missing_binary_errors as e:  # Tesseract/Poppler absent at call time (matched FIRST)
            raise AnonymizationDependencyError(_MISSING_IMAGE_REDACTOR_MSG) from e
        except ir.unprocessable_errors as e:  # supported magic the loader/redactor could not decode
            raise DocExampleBinaryError(
                f'{media} asset could not be decoded/redacted ({type(e).__name__}); refusing to egress un-scrubbed bytes (fail-closed).'
            ) from e


# --------------------------------------------------------------------------- #
# Base64-content-aware scrub (public: the single source of truth for callers    #
# that carry text as base64, e.g. doc-example content_b64 — SDK + plugin share) #
# --------------------------------------------------------------------------- #


def scrub_content_b64_object(anonymizer: Anonymizer, obj: dict[str, Any], *, content_field: str = 'content_b64') -> dict[str, Any]:
    """Anonymize an object that carries text as base64 in ``content_field``, in place.

    Reaches INSIDE the base64 (redact-before-egress): decode → anonymize the decoded
    text with ``anonymizer`` (pass the SAME scrubber used for the rest of a run so
    pseudonyms stay consistent) → re-encode. Every other field is anonymized with the
    generic tree walk. The decoded content goes through :meth:`Anonymizer.anonymize_bytes`:
    UTF-8 text gets the free-text scrub, a binary asset (image/PDF) the best-effort binary
    scrub, and an unsupported/corrupt asset or a missing binary dep fails closed
    (:class:`DocExampleBinaryError` / :class:`AnonymizationDependencyError`)
    rather than leave the content un-anonymized. Returns ``obj``.

    The SDK CLI scrubs each doc-example file with :meth:`Anonymizer.anonymize_bytes` directly,
    so a caller that carries the same content as base64 gets the same scrub here.
    """
    for field_name, value in list(obj.items()):
        if field_name == content_field and isinstance(value, str):
            try:
                raw = base64.b64decode(value, validate=True)
            except (binascii.Error, ValueError) as e:
                raise DocExampleBinaryError(f'{content_field!r} is not valid base64; cannot anonymize.') from e
            obj[field_name] = base64.b64encode(anonymizer.anonymize_bytes(raw, media_hint=obj.get('path'))).decode('ascii')
        else:
            obj[field_name] = anonymizer.anonymize_obj(value)
    return obj


# --------------------------------------------------------------------------- #
# Parallel file anonymization                                                  #
# --------------------------------------------------------------------------- #

#: Per-worker anonymizer, built once by the pool initializer (never shared/pickled).
_WORKER_ANONYMIZER: Anonymizer | None = None


def _worker_init(spec: AnonymizerSpec) -> None:
    """Pool initializer: build the anonymizer (incl. the spaCy model) ONCE per worker."""
    global _WORKER_ANONYMIZER
    _WORKER_ANONYMIZER = Anonymizer.from_spec(spec)


def _worker_run(path: str) -> str:
    """Pool task: rewrite one file using the worker's pre-built anonymizer."""
    anonymizer = _WORKER_ANONYMIZER
    if anonymizer is None:  # pragma: no cover - the initializer always runs first
        raise RuntimeError('worker Anonymizer was not initialized')
    anonymizer.anonymize_file(path)
    return path


def anonymize_files(paths: Iterable[str | Path], spec: AnonymizerSpec, *, max_workers: int | None = None) -> list[str]:
    """Anonymize many extractor-output files in parallel, rewriting each in place.

    Each worker builds its own engines ONCE (via the pool initializer) and the keyed
    hash is a pure function of ``value + key``, so there is **no shared in-memory
    map** and pseudonyms stay consistent across files and workers. Returns the
    processed paths.
    """
    tasks = [str(Path(p)) for p in paths]
    if not tasks:
        return []
    # Pre-flight in the PARENT so a missing dep raises the actionable error here,
    # not as an opaque BrokenProcessPool from a worker.
    Anonymizer.from_spec(spec)
    # Mint the key in the parent so workers only ever read it (no mint race).
    load_or_create_key(spec.workspace_root)
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init, initargs=(spec,)) as executor:
        return list(executor.map(_worker_run, tasks))


__all__ = [
    'FINGERPRINT_LEN',
    'AnonymizationDependencyError',
    'Anonymizer',
    'AnonymizerSpec',
    'DocExampleBinaryError',
    'KeyFingerprintMismatchError',
    'TreeHooks',
    'anonymize_files',
    'build_analyzer',
    'build_anonymizer',
    'key_fingerprint',
    'keyed_hash',
    'load_or_create_key',
    'scrub_content_b64_object',
    'verify_key_fingerprint',
]
