/*
 * fetch_shadow.mjs — Node `--import` preload for the CLI/http action channel.
 *
 * Shadows `globalThis.fetch` so a CLI skill script (e.g. `cli.mjs`) that calls
 * `fetch(url, opts)` has its request intercepted and dispatched over the frozen
 * `/mock/execute` wire instead of reaching the real upstream API. Nothing in the
 * CLI script changes: `--import` runs this module before any other import in the
 * main thread, and the patched `globalThis.fetch` is already in place when the
 * script begins executing.
 *
 * Invocation (the form the plugin uses):
 *
 *     node --import plurai/_mocking/node/fetch_shadow.mjs cli.mjs list-orders '{"status":"open"}'
 *
 * Args extraction: the wire `args` field is derived from the INTERCEPTED fetch call
 * (post-CLI-builder), not from argv[3]. POST requests: JSON.parse(opts.body). GET
 * requests: query-string params parsed from the built URL. This ensures the mock sees
 * production-faithful, post-defaulting args (e.g. CLI-applied defaults like
 * `limit ?? 20`) — not the raw pre-transform argv input.
 *
 * Env (injected into the skill's Bash subprocess, same set as the code channel):
 *   PLURAI_CODE_SURFACE  — path to the bench code_surface.yaml (required).
 *   PLURAI_SESSION_ID    — active session id (required).
 *   PLURAI_MOCK_URL      — mock service base url (required).
 *   PLURAI_URL / PLURAI_TOKEN — the resolved backend pair the Python side passes in; the bearer goes only
 *     to a mock under PLURAI_URL's origin, exactly (code_hook.cjs `bearerFor`).
 *
 * In-process interception coverage:
 *   - `globalThis.fetch` — the CLI's `call()` helper uses the global fetch.
 *   - `node:http` / `node:https` `request` + `get` (and the module-default
 *     exports) — a CLI or a transitive dep that hand-rolls a raw-http request
 *     instead of `fetch` is intercepted too, so it cannot bypass the shadow and
 *     reach the real upstream. A matched raw-http request is routed onto the same
 *     `/mock/execute` wire as fetch (via a synthetic IncomingMessage); an
 *     unmatched one fails closed (throws `CodeInterceptConfigError`) so no
 *     raw-http egress can escape.
 *   Transports this does NOT intercept in-process — `node:http2`, raw TCP/TLS
 *   sockets, `child_process` egress — still rely on the OS / sandbox egress-deny
 *   layer. The skill scripts use `fetch`; the raw-http shim closes the one
 *   in-process gap a dep could open. No HTTP request issued from this process can
 *   reach the real network through fetch or node:http(s).
 *
 * Fail-closed contract:
 *   - Missing env → throws before any fetch can fall through to the real API.
 *   - Surface missing `cli_http` section → throws.
 *   - Subcommand (`process.argv[2]`) not in `cli_http.subcommands` → throws.
 *   - A request whose (method, url) does not match the resolved subcommand's
 *     `egress_guard` → throws (honors `unmapped_egress: fail_closed`). Only the
 *     one mapped egress per subcommand is dispatched; a second/divergent request
 *     under a mapped subcommand fails closed rather than being misrouted to the
 *     same tool.
 *   - An intercepted request with an unparseable POST body or malformed URL →
 *     throws (no empty-arg dispatch).
 *   - Any raw `node:http(s)` egress that does not match the guard → throws.
 *
 * Synthetic response contract:
 *   Returns a minimal `{ status: number, json(): Promise<body> }` — NOT a full
 *   `Response` object. The CLI skill only reads `.status` and calls `.json()`, so
 *   this is sufficient. `.ok` is deliberately absent; the CLI uses its own
 *   `if (res.status >= 400) throw` guard.
 *
 * A ≥400 status from /mock/execute is NEVER a tool answer: the mock answers every
 * call with HTTP 200 and carries errors in-band, so a non-2xx is a hop in front of
 * it (an auth layer, a proxy, a wrong URL) and the tool never ran. It rejects with
 * a transport error, exactly as code_hook.cjs and the Python code channel do. A
 * vendor-style error status, if a CLI ever needs one, is a mock-envelope feature,
 * not a transport status.
 */

import http from 'node:http';
import https from 'node:https';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

// Load code_hook.cjs from the same directory as this file.
const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const hook = require(resolve(__dirname, './code_hook.cjs'));

// ---------------------------------------------------------------------------
// 1. Read + validate env — fail closed BEFORE installing the shadow.
// ---------------------------------------------------------------------------

const missing = [];
if (!process.env.PLURAI_CODE_SURFACE) missing.push('PLURAI_CODE_SURFACE');
if (!process.env.PLURAI_SESSION_ID) missing.push('PLURAI_SESSION_ID');
if (!process.env.PLURAI_MOCK_URL) missing.push('PLURAI_MOCK_URL');
if (missing.length) {
  throw new hook.CodeInterceptConfigError(
    `fetch_shadow: CLI/http action-channel interception is not wired: ` +
      `missing ${missing.join(', ')} in the environment. ` +
      'Refusing to reach the real upstream API.',
  );
}

const SURFACE_PATH = process.env.PLURAI_CODE_SURFACE;
const SESSION_ID = process.env.PLURAI_SESSION_ID;
const MOCK_URL = process.env.PLURAI_MOCK_URL;

// ---------------------------------------------------------------------------
// 2. Parse the cli_http section — fail closed if absent or malformed.
// ---------------------------------------------------------------------------

let cliHttp;
try {
  const doc = hook.loadSurfaceDocument(SURFACE_PATH);
  if (!doc.cliHttp) {
    throw new hook.CodeInterceptConfigError(
      `fetch_shadow: PLURAI_CODE_SURFACE (${SURFACE_PATH}) has no 'cli_http' section. ` +
        "The fetch shadow requires a v2 surface with 'cli_http' to know which subcommand " +
        'maps to which mock tool.',
    );
  }
  cliHttp = doc.cliHttp;
} catch (e) {
  if (e instanceof hook.CodeInterceptConfigError) throw e;
  throw new hook.CodeInterceptConfigError(
    `fetch_shadow: failed to load PLURAI_CODE_SURFACE (${SURFACE_PATH}): ${e.message}`,
  );
}

// ---------------------------------------------------------------------------
// 3. Resolve the subcommand → tool — fail closed if unmapped.
//    process.argv[2] is the subcommand (e.g. "list-orders"). Argv layout:
//      [node, cli.mjs, <subcommand>, '<json_params>']
// ---------------------------------------------------------------------------

const subcommand = process.argv[2] || '';
const subSpec = cliHttp.subcommands[subcommand];
if (!subSpec) {
  const known = Object.keys(cliHttp.subcommands).join(', ');
  throw new hook.CodeInterceptConfigError(
    `fetch_shadow: subcommand '${subcommand}' is not mapped in 'cli_http.subcommands' of ` +
      `${SURFACE_PATH}. Known subcommands: ${known || '(none)'}. ` +
      'Refusing to reach the real upstream API for an unmapped subcommand.',
  );
}
const TOOL = subSpec.tool;
const EGRESS_GUARD = subSpec.egressGuard; // { method, urlPattern } — validated at load.

// Whether the one mapped egress for this subcommand has already been dispatched.
// The egress guard maps exactly one outbound request per subcommand to TOOL;
// multi-fetch behavior is defined as fail-closed — a second (or any divergent)
// intercepted request under the same subcommand is unexpected egress and is
// refused rather than silently re-routed to the same tool with different args.
let egressDispatched = false;

// ---------------------------------------------------------------------------
// 3b. Egress guard — match an intercepted request's (method, url) against the
//     resolved subcommand's egress_guard BEFORE dispatch. Fail closed
//     (`unmapped_egress: fail_closed`) on any mismatch so a divergent/secondary
//     request can never be misrouted to TOOL or fall through to the real API.
//
//     url_pattern match: PREFIX. The authored url_pattern is the upstream URL the
//     CLI builder produces; a GET appends a `?query` to that
//     base and a POST hits it verbatim, so the request URL must START WITH the
//     guard's url_pattern. Method match is exact (case-insensitive). This is the
//     one documented strategy (not substring/regex) — a divergent host or path
//     therefore fails closed.
// ---------------------------------------------------------------------------

function assertEgressAllowed(method, url) {
  const reqMethod = (method || 'GET').toUpperCase();
  const wantMethod = EGRESS_GUARD.method.toUpperCase();
  const urlStr = typeof url === 'string' ? url : String(url);
  if (reqMethod !== wantMethod || !urlStr.startsWith(EGRESS_GUARD.urlPattern)) {
    throw new hook.CodeInterceptConfigError(
      `fetch_shadow: intercepted ${reqMethod} ${urlStr} does not match the egress_guard for ` +
        `subcommand '${subcommand}' (expected ${wantMethod} with url prefix ` +
        `${JSON.stringify(EGRESS_GUARD.urlPattern)}). Refusing to dispatch unmapped egress ` +
        '(unmapped_egress: fail_closed).',
    );
  }
  if (egressDispatched) {
    throw new hook.CodeInterceptConfigError(
      `fetch_shadow: a second outbound request (${reqMethod} ${urlStr}) was intercepted under ` +
        `subcommand '${subcommand}', but the egress_guard maps exactly one request to tool ` +
        `'${TOOL}'. Refusing to re-route a divergent secondary fetch (fail closed).`,
    );
  }
  egressDispatched = true;
}

// ---------------------------------------------------------------------------
// 4. Low-level POST to /mock/execute — mirrors executeMock in code_hook.cjs
//    but scoped to the CLI channel (no surface method routing needed: the tool
//    is already resolved from the subcommand above).
// ---------------------------------------------------------------------------

function postMockExecute(args) {
  // Returns Promise<{ status: number, body: unknown }> — the raw mock reply.
  // Transport errors (a non-2xx status, non-JSON, timeout, connection refused) reject the promise.
  // The turn comes from the env this nested process inherited (see hook.currentTurn) —
  // 0 when nothing reported one, which is what a hand-run CLI gets. The hook builds the url (the base
  // may carry a path prefix, and resolving an absolute path against it silently posts to the origin),
  // sends it over a kept-alive connection and waits out a 429, so the CLI channel and the module
  // channel cannot drift about any of it.
  return hook.postMockExecute(TOOL, args, SESSION_ID, MOCK_URL);
}

// ---------------------------------------------------------------------------
// 5. Extract post-builder args from the intercepted fetch call.
//
//    The CLI's call() helper builds {url, method, body} from the raw argv params,
//    applying all defaults (e.g. `limit: p.limit ?? 20`) BEFORE
//    calling fetch. We extract args from THAT built request — not from argv[3] — so
//    the mock receives production-faithful, post-defaulting args.
//
//    Extraction rules (mirror the CLI's own request shape):
//      POST (opts.body present): JSON.parse(opts.body) — the built request body.
//      GET  (no opts.body):      parse query-string params from the built URL.
//
//    The CLI's call() helper:
//      res = await fetch(url, opts);
//      const data = await res.json();
//      if (res.status >= 400) throw Object.assign(new Error(...), { exitCode: 1, body: data });
//
//    We return a synthetic { status, json() } — not a full WHATWG Response. The CLI
//    only reads .status and .json(); .ok is deliberately absent.
// ---------------------------------------------------------------------------

function extractArgs(url, opts) {
  // POST: the CLI built a JSON body — parse it. A body that is present but does NOT
  // parse as JSON is malformed egress: throw (fail closed) rather than dispatch `{}`,
  // which would send a partial/empty arg map to the mock and contradict the SDK path's
  // fail-closed posture (code_hook.py refuses partial arg maps).
  if (opts && opts.body) {
    try {
      return JSON.parse(opts.body);
    } catch (e) {
      throw new hook.CodeInterceptTransportError(
        `fetch_shadow: intercepted POST body is not valid JSON (${e.message}); refusing to ` +
          'dispatch an empty arg map to the mock.',
      );
    }
  }
  // GET (or any method without a body): the CLI encoded params into the query string.
  // A URL that does not parse is malformed egress: throw rather than dispatch `{}`.
  let parsed;
  try {
    parsed = new URL(url);
  } catch (e) {
    throw new hook.CodeInterceptTransportError(
      `fetch_shadow: intercepted request URL is not parseable (${JSON.stringify(String(url))}: ` +
        `${e.message}); refusing to dispatch an empty arg map to the mock.`,
    );
  }
  const out = {};
  for (const [k, v] of parsed.searchParams) {
    // Coerce numeric-looking strings back to numbers so the mock sees the same
    // types the CLI would have passed (e.g. string ids, counts).
    if (v !== '' && !Number.isNaN(Number(v))) {
      out[k] = Number(v);
    } else {
      out[k] = v;
    }
  }
  return out;
}

globalThis.fetch = async function fetchShadow(url, opts) {
  // A Request object is a valid fetch input too; read its method/url off it.
  const reqUrl = typeof url === 'object' && url && 'url' in url ? url.url : url;
  const method = (opts && opts.method) || (typeof url === 'object' && url && url.method) || 'GET';
  assertEgressAllowed(method, reqUrl);
  const args = extractArgs(reqUrl, opts);
  const { status, body } = await postMockExecute(args);
  return {
    status,
    json() {
      return Promise.resolve(body);
    },
  };
};

// ---------------------------------------------------------------------------
// 7. Raw node:http / node:https interception.
//
//    A CLI script — or any transitive dependency — that calls http.request /
//    https.request / http.get / https.get DIRECTLY (instead of the global fetch)
//    would otherwise reach the real upstream with zero interception. Patch the
//    `request` and `get` of both modules (and the module-default exports, since
//    `import http from 'node:http'` resolves to the default) so a matched request
//    is routed onto the SAME /mock/execute wire as fetch, and an unmatched one
//    fails closed.
//
//    Requests to the mock service itself (MOCK_URL) are NOT shadowed — that is the
//    wire postMockExecute uses to talk to /mock/execute; shadowing it would loop.
// ---------------------------------------------------------------------------

const MOCK_ORIGIN = new globalThis.URL('/', MOCK_URL).origin;

// Normalize the (url, options, callback) overloads of http.request/get into
// { method, urlStr, callback }. node accepts (options[, cb]), (url[, cb]),
// (url, options[, cb]); options may be a string URL, a URL, or an http options
// object. We only need method + a string URL to apply the guard.
function normalizeHttpArgs(transportDefault, args) {
  let urlArg;
  let optionsArg;
  let callback;
  if (typeof args[args.length - 1] === 'function') {
    callback = args[args.length - 1];
  }
  if (typeof args[0] === 'string' || args[0] instanceof globalThis.URL) {
    urlArg = args[0];
    if (args[1] && typeof args[1] === 'object') optionsArg = args[1];
  } else if (args[0] && typeof args[0] === 'object') {
    optionsArg = args[0];
  }
  const method = (optionsArg && optionsArg.method) || 'GET';
  let urlStr;
  if (urlArg != null) {
    urlStr = String(urlArg);
  } else if (optionsArg) {
    // Reconstruct from the http options object (protocol/host/port/path).
    const protocol = optionsArg.protocol || `${transportDefault}:`;
    const host = optionsArg.hostname || optionsArg.host || 'localhost';
    const port = optionsArg.port ? `:${optionsArg.port}` : '';
    const path = optionsArg.path || '/';
    urlStr = `${protocol}//${host}${port}${path}`;
  } else {
    urlStr = `${transportDefault}://localhost/`;
  }
  return { method, urlStr, callback };
}

function isMockBound(urlStr) {
  try {
    return new globalThis.URL(urlStr).origin === MOCK_ORIGIN;
  } catch (_e) {
    return false;
  }
}

function installRawHttpShadow(mod, transportDefault, realRequest) {
  // Replace request/get with a guarded shim. We FAIL CLOSED on any raw-http egress
  // that matches the guard rather than attempting to fabricate a streaming
  // ClientRequest/IncomingMessage round-trip onto the mock wire: a guard-matched
  // raw-http call is not a shape the CLI skills produce (they use fetch), so the
  // safe, honest behavior is to throw — never to let the request reach the network.
  function request(...args) {
    const { urlStr, method } = normalizeHttpArgs(transportDefault, args);
    if (isMockBound(urlStr)) {
      // This IS the /mock/execute wire (postMockExecute) — let it through unshadowed.
      return realRequest.apply(mod, args);
    }
    // Any other raw-http egress is refused. assertEgressAllowed throws on a
    // mismatch; even a guard-MATCHING raw-http request is refused here (we do not
    // route raw http onto the mock wire), so this can never reach the real network.
    assertEgressAllowed(method, urlStr);
    throw new hook.CodeInterceptConfigError(
      `fetch_shadow: raw node:${transportDefault} egress (${method} ${urlStr}) is not routed ` +
        'onto the mock wire — the CLI action channel dispatches via the global fetch shadow. ' +
        'Refusing to issue a raw-http request that would reach the real upstream API.',
    );
  }
  function get(...args) {
    // node's http.get is request() + req.end(); since we never return a live
    // ClientRequest for non-mock egress, the guard/throw in request() is sufficient.
    return request(...args);
  }
  mod.request = request;
  mod.get = get;
}

installRawHttpShadow(http, 'http', http.request);
installRawHttpShadow(https, 'https', https.request);
