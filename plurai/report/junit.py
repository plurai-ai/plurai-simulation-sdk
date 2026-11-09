"""``junit.xml``: one test case per scenario × policy.

A failed cell is a ``failure``. A gating cell that could not be measured is an ``error``, because the check does not
pass over it. Unstable and advisory cells pass and carry their outcome in ``system-out``. Not applicable cells and
advisory cells that could not be measured are ``skipped``.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from plurai.report.render import class_label, not_measured_reason, why_text
from plurai.report.run import Run, scrub_surrogates

_XML_ILLEGAL = re.compile('[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]')


def _clean(text: str) -> str:
    return _XML_ILLEGAL.sub('', scrub_surrogates(text))


def render_junit(run: Run, *, include_content: bool = False) -> str:
    """The run as a JUnit XML document. A failure carries judge text only when ``include_content`` is on, so a caller that
    does not apply the visibility rule leaks nothing."""
    suite = ET.Element('testsuite', name='plurai simulation')
    failures = errors = skipped = 0
    for cell in run.cells:
        result = cell.outcome
        case = ET.SubElement(suite, 'testcase', classname=_clean(cell.scenario), name=_clean(cell.policy))
        if result.outcome == 'failed':
            failures += 1
            failure = ET.SubElement(case, 'failure', message=result.detail)
            failure.text = _clean(why_text(cell, include_content=include_content, fallback='')) or None
        elif result.outcome == 'not_measured':
            reason = _clean(not_measured_reason(cell))
            if result.level == 'gating':
                errors += 1
                ET.SubElement(case, 'error', type=class_label(result.classes), message=reason)
            else:
                skipped += 1
                ET.SubElement(case, 'skipped', message=f'not measured, advisory: {reason}')
        elif result.outcome == 'not_applicable':
            skipped += 1
            ET.SubElement(case, 'skipped', message='not applicable')
        elif result.outcome in ('unstable', 'advisory'):
            ET.SubElement(case, 'system-out').text = f'{result.outcome}: {result.detail}'
    suite.set('tests', str(len(run.cells)))
    suite.set('failures', str(failures))
    suite.set('errors', str(errors))
    suite.set('skipped', str(skipped))
    ET.indent(suite)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(suite, encoding='unicode') + '\n'
