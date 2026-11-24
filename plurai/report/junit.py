"""``junit.xml``: one test case per scenario × policy, one test suite per scenario.

Each scenario is a ``testsuite`` and each of its policies a ``testcase``. A failed gating scenario × policy is a ``failure``
whose message is the outcome and then the judge's reason (the same reason fills its body); a passed one is a pass. A gating
scenario × policy that was not measured is an ``error`` with its reason as the message: a reader that treats ``skipped`` as
green would otherwise show a run that measured nothing as clean. Every other case is ``skipped`` with its reason as the
message: an advisory scenario × policy that was not measured, one that does not apply, one deferred to a sibling policy, one a
``ci.yaml`` rule ignores (whatever it computed) and each one of a scenario that was not run. Unstable cells pass and carry
their outcome in ``system-out``; so does an advisory cell that failed, which the check does not count, and the text says it is
advisory. The check itself is the job's exit code, not this file.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from plurai.report.render import deferred_text, not_measured_reason, why_text
from plurai.report.run import Ignored, Run, scrub_surrogates

_XML_ILLEGAL = re.compile('[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]')


def _clean(text: str) -> str:
    return _XML_ILLEGAL.sub('', scrub_surrogates(text))


def _skipped(case: ET.Element, message: str) -> None:
    ET.SubElement(case, 'skipped', message=message)


def _ignored_message(ignored: Ignored, *, prefix: str = 'ignored by ci.yaml') -> str:
    until = f' (expires {ignored.expires.isoformat()})' if ignored.expires else ''
    return _clean(f'{prefix}: {ignored.reason}{until}')


def _count(suite: ET.Element, tests: int, failures: int, errors: int, skipped: int) -> None:
    suite.set('tests', str(tests))
    suite.set('failures', str(failures))
    suite.set('errors', str(errors))
    suite.set('skipped', str(skipped))


def render_junit(run: Run, *, include_content: bool = False) -> str:
    """The run as a JUnit XML document. A failure carries judge text only when ``include_content`` is on, so a caller that
    does not apply the visibility rule leaks nothing."""
    root = ET.Element('testsuites', name='plurai simulation')
    suites: dict[str, ET.Element] = {}
    #: Per suite: tests, failures, errors, skipped.
    totals: dict[str, list[int]] = {}

    def case_of(scenario: str, policy: str) -> ET.Element:
        name = _clean(scenario)
        if name not in suites:
            suites[name] = ET.SubElement(root, 'testsuite', name=name)
            totals[name] = [0, 0, 0, 0]
        totals[name][0] += 1
        return ET.SubElement(suites[name], 'testcase', classname=name, name=_clean(policy))

    def skip(case: ET.Element, scenario: str, message: str) -> None:
        totals[_clean(scenario)][3] += 1
        _skipped(case, message)

    for cell in run.cells:
        result = cell.outcome
        case = case_of(cell.scenario, cell.policy)
        if cell.ignored is not None:
            skip(case, cell.scenario, _ignored_message(cell.ignored))
        elif result.outcome == 'failed' and result.level == 'gating':
            totals[_clean(cell.scenario)][1] += 1
            why = _clean(why_text(cell, include_content=include_content, fallback=''))
            # A reader that renders a test tree prints the message and not the body, so the message carries the reason too.
            failure = ET.SubElement(case, 'failure', message=f'{result.detail}\n{why}' if why else result.detail)
            failure.text = why or None
        elif result.outcome == 'not_measured' and result.level == 'advisory':
            skip(case, cell.scenario, f'not measured, advisory: {_clean(not_measured_reason(cell))}')
        elif result.outcome == 'not_measured':
            totals[_clean(cell.scenario)][2] += 1
            ET.SubElement(case, 'error', message=f'not measured: {_clean(not_measured_reason(cell))}')
        elif result.outcome == 'not_applicable':
            skip(case, cell.scenario, 'not applicable')
        elif result.outcome == 'deferred':
            skip(case, cell.scenario, _clean(deferred_text(cell)))
        elif result.outcome in ('failed', 'unstable'):
            level = ' (advisory)' if result.level == 'advisory' else ''
            ET.SubElement(case, 'system-out').text = f'{result.outcome}{level}: {result.detail}'
    for skipped_scenario in run.not_run:
        message = '; '.join(_ignored_message(rule, prefix='not run, ignored by ci.yaml') for rule in skipped_scenario.ignored)
        for policy in skipped_scenario.policies:
            skip(case_of(skipped_scenario.scenario, policy), skipped_scenario.scenario, message)
    for name, suite in suites.items():
        _count(suite, *totals[name])
    _count(root, *(sum(t[i] for t in totals.values()) for i in range(4)))
    ET.indent(root)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding='unicode') + '\n'
