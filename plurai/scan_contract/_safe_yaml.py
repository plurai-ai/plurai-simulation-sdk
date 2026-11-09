"""The one place the backend parses YAML: a load whose failure never carries the document's text.

A ``yaml.YAMLError`` quotes the document it failed on: its message, its ``problem``/``context``
prose, and the snippet its marks print. Any handler that logs, reports or re-raises that error with
its text (``exc_info=True``, ``logger.exception``, ``f'{exc}'``, a job report) therefore writes
customer config — ``policies_config.yaml``, ``scenario_variables.yaml`` — wherever that goes.

:func:`safe_yaml_load` is ``yaml.safe_load`` whose parse failure is a :class:`ConfigParseError`
carrying only the error's kind and line/column, with nothing chained. Every YAML read in the backend
goes through it; a security test fails on any raw ``yaml`` load (or compose/parse/scan, or a
loader class) anywhere else, with no exceptions list.

pyyaml and the stdlib only: the scan-contract package imports this module and also runs in
client environments, where nothing else is available.
"""

from __future__ import annotations

from typing import IO, Any

import yaml


class ConfigParseError(yaml.YAMLError):
    """A YAML parse failure with the document's content stripped.

    ``str()`` is :func:`yaml_error_summary` of the original error — its type and line/column,
    nothing else, and there are no marks on it to print. It subclasses ``yaml.YAMLError`` so every
    ``except yaml.YAMLError`` handler still catches it.
    """


def _mark_location(mark: yaml.Mark) -> str:
    """``line N, column N`` (1-indexed) off a ``yaml.Mark`` — NEVER the mark itself:
    ``Mark.__str__``/``get_snippet()`` prints the document text around the point."""
    return f'line {mark.line + 1}, column {mark.column + 1}'


def yaml_error_summary(exc: yaml.YAMLError) -> str:
    """The error's type and ``problem_mark``/``context_mark`` positions — never its description
    text, which is pyyaml's prose about the document and can quote a found token verbatim."""
    if isinstance(exc, ConfigParseError):
        return str(exc)
    marks = [_mark_location(mark) for mark in (getattr(exc, 'problem_mark', None), getattr(exc, 'context_mark', None)) if mark is not None]
    return f'{type(exc).__name__} at {"; ".join(marks)}' if marks else type(exc).__name__


def safe_yaml_load(stream: str | bytes | IO[str] | IO[bytes]) -> Any:  # noqa: ANN401 - arbitrary YAML document
    """``yaml.safe_load(stream)``, but a parse failure raises :class:`ConfigParseError`.

    Raised OUTSIDE the ``except`` block on purpose: ``raise ... from None`` only hides
    ``__context__`` from a traceback printer, while the object still holds the original error and
    its document buffer. Raising after the block leaves nothing chained at all.
    """
    try:
        return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        summary = yaml_error_summary(exc)
    raise ConfigParseError(summary)


def safe_yaml_compose(text: str) -> yaml.Node | None:
    """``yaml.compose(text)`` (the node tree, with the marks an in-place edit splices at), but a parse failure raises
    :class:`ConfigParseError`."""
    try:
        return yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        summary = yaml_error_summary(exc)
    raise ConfigParseError(summary)


def bare_flow_keys(text: str) -> list[tuple[int, int]]:
    """``(line, column)`` (1-indexed) of every flow-mapping key written with no ``:`` and no value.

    ``{description: numeric column to average, qualified with its dataset, optional: false}`` loads
    clean: the unquoted comma ends the string, and its tail becomes a key whose value is null. The
    loaded data cannot tell that from a deliberate null, so this reads the composed nodes: a flow-mapping
    scalar key whose value is the implicit empty null and whose text is not followed by ``:``. An
    explicit ``key:`` / ``key: ~``, a quoted comma and a flow sequence are not reported.

    Positions only, never text, by this module's rule. ``[]`` for a document that does not parse:
    the load reports that.
    """
    found: list[tuple[int, int]] = []
    seen: set[int] = set()  # an alias re-enters a composed node; a recursive one would never end

    def walk(node: yaml.Node) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                if (
                    node.flow_style
                    and isinstance(key, yaml.ScalarNode)
                    and isinstance(value, yaml.ScalarNode)
                    and value.tag == 'tag:yaml.org,2002:null'
                    and value.value == ''
                    and not text[key.end_mark.index :].lstrip().startswith(':')
                ):
                    found.append((key.start_mark.line + 1, key.start_mark.column + 1))
                walk(key)
                walk(value)
        elif isinstance(node, yaml.SequenceNode):
            for item in node.value:
                walk(item)

    try:
        for document in yaml.compose_all(text, Loader=yaml.SafeLoader):
            if document is not None:
                walk(document)
    except yaml.YAMLError:
        return []
    return found


class YamlBoundsError(ConfigParseError):
    """A document refused for its size or its alias expansion, never for its content.

    ``cause`` is a stable code: ``too_large`` (bytes), ``expansion`` (nodes after aliases expand),
    ``recursive_alias`` or ``too_deep``. The message states the cause and the limit and quotes nothing.
    """

    def __init__(self, cause: str, message: str) -> None:
        super().__init__(message)
        self.cause = cause


def _expanded_size(root: yaml.Node, max_nodes: int) -> None:
    """Raise when ``root`` holds more than ``max_nodes`` nodes once every alias is expanded.

    An alias is a reference to the anchored node, so composing a billion-laughs document is cheap and
    constructing it is not. The expanded size is computed on the shared graph, once per node, before
    anything is constructed. A node that contains itself is refused.
    """
    sizes: dict[int, int] = {}
    open_nodes: set[int] = set()
    stack: list[tuple[yaml.Node, bool]] = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        ident = id(node)
        if expanded:
            open_nodes.discard(ident)
            sizes[ident] = 1 + sum(sizes[id(child)] for child in _children(node))
            if sizes[ident] > max_nodes:
                raise YamlBoundsError('expansion', f'the document expands to more than {max_nodes} nodes once its aliases are resolved')
            continue
        if ident in sizes:
            continue
        if ident in open_nodes:
            raise YamlBoundsError('recursive_alias', 'the document contains an alias that refers to its own ancestor')
        open_nodes.add(ident)
        stack.append((node, True))
        stack.extend((child, False) for child in _children(node) if id(child) not in sizes)


def _children(node: yaml.Node) -> list[yaml.Node]:
    if isinstance(node, yaml.SequenceNode):
        return list(node.value)
    if isinstance(node, yaml.MappingNode):
        return [part for pair in node.value for part in pair]
    return []


def safe_yaml_load_bounded(text: str, *, max_bytes: int, max_nodes: int) -> Any:  # noqa: ANN401 - arbitrary YAML document
    """:func:`safe_yaml_load` for untrusted text: refuse an oversized document or an alias bomb before constructing it.

    Raises :class:`YamlBoundsError` for a limit and :class:`ConfigParseError` for a parse failure; neither
    carries the document's text.
    """
    if len(text.encode('utf-8')) > max_bytes:
        raise YamlBoundsError('too_large', f'the document is larger than {max_bytes} bytes')
    loader = yaml.SafeLoader(text)
    summary: str | None = None
    too_deep = False
    try:
        node = loader.get_single_node()
        if node is None:
            return None
        _expanded_size(node, max_nodes)
        return loader.construct_document(node)
    except yaml.YAMLError as exc:
        if isinstance(exc, YamlBoundsError):
            raise
        summary = yaml_error_summary(exc)
    except RecursionError:
        too_deep = True
    finally:
        loader.dispose()
    if too_deep:
        raise YamlBoundsError('too_deep', 'the document is nested too deeply')
    raise ConfigParseError(summary)
