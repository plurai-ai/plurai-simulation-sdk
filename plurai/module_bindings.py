"""The names a ``simulate.py`` binds at module scope: one predicate, shared by three repos.

The walk lives in its own module because it uses ``field`` as a loop variable, which in ``orchestrator.py``
would shadow that module's ``from dataclasses import field`` (ruff F402). The code is identical to the plugin's
two copies of ``_module_scope_bindings``, and their drift test compares the three as ASTs with docstrings dropped.
"""

from __future__ import annotations

import ast


def _module_scope_bindings(tree: ast.Module) -> set[str]:
    """Every name simulate.py binds AT IMPORT TIME — the set ``getattr(module, name)`` can hit.

    The question :func:`_find_function` answers NARROWLY, asked the way the runner asks it. The
    child resolves every hook with ``getattr`` on the imported module, so ``reset = _impl`` and
    ``from ._stores import reset`` export the hook exactly as a ``def`` does; a check that saw
    only ``def`` would refuse a harness the runner drives perfectly well, and a false refusal is
    the reason an author stops reading the verifier. Statement bodies that RUN at import
    (``if`` / ``try`` / ``with`` / ``for``) are descended into; function and class bodies are not,
    because a name bound there is not an attribute of the module.

    A deliberate copy: the plugin's ``verify_simulate.py`` and ``verify_config.py`` each keep the same
    function rather than import this one, and the SDK needs the same answer without depending on the plugin.
    The three must answer the binding question identically; their docstrings are free to differ.
    """
    bound: set[str] = set()

    def _walk(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
                continue  # its body is a different scope
            if isinstance(node, ast.Assign):
                targets: list[ast.expr] = []
                for target in node.targets:
                    if isinstance(target, (ast.Tuple, ast.List)):
                        targets.extend(target.elts)
                    else:
                        targets.append(target)
                for target in targets:
                    if isinstance(target, ast.Name):
                        bound.add(target.id)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                if isinstance(node.target, ast.Name):
                    bound.add(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    bound.add(alias.asname or alias.name.split('.')[0])
            for field in ('body', 'orelse', 'finalbody'):
                nested = getattr(node, field, None)
                if isinstance(nested, list):
                    _walk([s for s in nested if isinstance(s, ast.stmt)])
            for handler in getattr(node, 'handlers', []) or []:
                _walk(handler.body)

    _walk(tree.body)
    return bound
