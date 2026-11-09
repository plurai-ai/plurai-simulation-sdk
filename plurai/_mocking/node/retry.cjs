'use strict';
// The retry arithmetic of the mock wire, the Node mirror of plurai/_http_retry.py.
//
// A hosted backend answers 429 when it throttles a caller, and every path that reaches it waits as
// `Retry-After` says (with jitter and a cap) and tries again. The numbers below are the Python
// module's; a parity test holds the two files to each other.

const RATE_LIMITED_STATUS = 429;
const RATE_LIMIT_RETRIES = 5;
const BACKOFF_BASE_SECONDS = 1.0;
const BACKOFF_CAP_SECONDS = 30.0;
// Retries after the first attempt that a transient failure (a 503, a 502 the proxy marks as never connected to the mock, a
// connection that failed while connecting) gets: the waits 1, 2, 4, 8, 16 and 30 seconds, about 61 to 66 in all, which
// outlasts the 35 seconds a backend upgrade rejects every request for.
const TRANSIENT_RETRIES = 6;

// The wait a Retry-After header value asks for, in seconds, or null when it asks for none. Both
// RFC 9110 forms: delta-seconds, and an HTTP-date read against `nowMs` (epoch milliseconds).
function parseRetryAfter(value, nowMs = Date.now()) {
  const raw = String(value ?? '').trim();
  if (!raw) return null;
  if (/^\d+$/.test(raw)) return Number(raw);
  const when = Date.parse(raw);
  if (Number.isNaN(when)) return null;
  return Math.max(0, (when - nowMs) / 1000);
}

// The wait before the retryNumber-th retry (counting from 1), in seconds: exponential with additive
// jitter, capped. The floor is the base and the schedule grows.
function backoffSeconds(retryNumber, base = BACKOFF_BASE_SECONDS, cap = BACKOFF_CAP_SECONDS, jitter = Math.random) {
  return Math.min(cap, base * 2 ** (retryNumber - 1) + jitter() * base);
}

// The wait before retrying a 429, in milliseconds: the larger of the back-off and what Retry-After
// asked, capped.
function rateLimitWaitMs(retryNumber, retryAfterHeader, base = BACKOFF_BASE_SECONDS, cap = BACKOFF_CAP_SECONDS) {
  const asked = parseRetryAfter(retryAfterHeader) ?? 0;
  return Math.min(cap, Math.max(backoffSeconds(retryNumber, base, cap), asked)) * 1000;
}

// What the backend's mock proxy says when its own connection to the mock failed, before it sent anything. Any other reason
// it names ("no answer within ...", "Server disconnected") followed a request the mock may have run.
const PROXY_CONNECT_FAILED = /^mock service for \S+ unreachable: Cannot connect to host\b/;

// The other answers a 503 can carry that waiting does not change: an env held for a human, the capacity refusal, and every
// credit kind (plurai/credit.py CREDIT_KINDS).
const STANDING_REFUSAL_CODES = [
  'awaiting_edit_finalize',
  'capacity_exhausted',
  'credit-refused',
  'credit-unavailable',
  'free-tier-ceiling',
  'insufficient-credit',
  'new-run-refused',
];

// The standing refusals that outrank a typed unavailable block on the same answer: the capacity refusal and every credit
// kind. An env held for a human does not, as in the Python client.
const REFUSAL_CODES_THAT_OUTRANK_UNAVAILABLE = STANDING_REFUSAL_CODES.filter((code) => code !== 'awaiting_edit_finalize');

// Connect-phase socket errors: the request never left, so a repeat cannot apply anything twice. A name that does not resolve
// (ENOTFOUND, EAI_AGAIN) is nobody's outage and never waits; a reset, a broken pipe or a hang-up may follow a sent request.
const CONNECT_FAILURE_CODES = new Set(['ECONNREFUSED']);
// No route to the host or network counts only while connecting: the same code on a read says nothing about whether the request was heard.
const NO_ROUTE_CODES = new Set(['EHOSTUNREACH', 'ENETUNREACH']);

// What the wire does with an answer to POST /mock/execute: 'wait' (repeat it under the transient budget), 'throttle' (repeat
// it under the 429 budget), 'outage' (the backend's typed unavailable block: never waited out), or 'raise' (take it as
// it is). Only what the mock cannot have run is repeated: a 503 the proxy answers before it forwards anything (unless its code
// names a standing refusal), and a 502 saying the proxy could not connect. Any other 502, and a 504, may follow a tool call that
// already ran: never re-sent. A shared table of cases holds this and the Python client to the same rule.
function replyVerdict(statusCode, text) {
  let body = null;
  try {
    body = JSON.parse(text);
  } catch (_e) {
    // an answer that is not JSON carries no code
  }
  const record = body && typeof body === 'object' ? body : {};
  // The backend's typed unavailable block, read by where it sits (never by its code): it already outlasted the backend's own
  // retries, so it is never waited out, whatever status carries it.
  // A credit or capacity code on the same answer wins, as it does in every other reader.
  if (record.unavailable && typeof record.unavailable === 'object' && !REFUSAL_CODES_THAT_OUTRANK_UNAVAILABLE.includes(record.code)) return 'outage';
  if (statusCode === RATE_LIMITED_STATUS) return 'throttle';
  if (statusCode === 503) return STANDING_REFUSAL_CODES.includes(record.code) ? 'raise' : 'wait';
  if (statusCode === 502) {
    // A standing refusal on the same answer outranks the proxy's "could not connect": waiting does not clear it.
    if (STANDING_REFUSAL_CODES.includes(record.code)) return 'raise';
    return typeof record.error === 'string' && PROXY_CONNECT_FAILED.test(record.error) ? 'wait' : 'raise';
  }
  return 'raise';
}

// What the wire does with a socket failure of POST /mock/execute: 'wait' only for a connection that failed while connecting.
// A timeout counts only when it fired before the socket connected (`connectTimeout`, set by the wire's own timer).
function failureVerdict(err) {
  const connecting =
    err && (CONNECT_FAILURE_CODES.has(err.code) || (NO_ROUTE_CODES.has(err.code) && err.syscall === 'connect') || err.connectTimeout === true);
  return connecting ? 'wait' : 'raise';
}

module.exports = {
  RATE_LIMITED_STATUS,
  RATE_LIMIT_RETRIES,
  BACKOFF_BASE_SECONDS,
  BACKOFF_CAP_SECONDS,
  TRANSIENT_RETRIES,
  STANDING_REFUSAL_CODES,
  replyVerdict,
  failureVerdict,
  parseRetryAfter,
  backoffSeconds,
  rateLimitWaitMs,
};
