'use strict';
/*
 * Node `code` action-channel import hook — the Node analog of plurai's Python
 * meta_path hook (plurai/_mocking/code_hook.py). Shadows the vendor module(s) named in a
 * `code_surface.yaml` so the skill's real Node code — `const resend = new Resend(key);
 * await resend.emails.send(...)` — runs unchanged, but each call is dispatched over the
 * frozen `/mock/execute` wire instead of reaching the real vendor API.
 *
 * Invocation (the form the plugin uses):
 *
 *     node --require <plurai>/_mocking/node/code_hook.cjs send.js
 *
 * `--require` runs this before the snippet, so the CommonJS `require('resend')` cache is
 * patched first. Because a snippet may be ESM (`import { Resend } from
 * 'resend'`), this file ALSO registers an ESM loader hook (Node >= 20.6 `module.register`)
 * so `import 'resend'` is shadowed too — one entrypoint covers both module systems.
 *
 * Driven ENTIRELY by the code_surface.yaml data (same schema as the Python hook):
 *   modules.<m>.{ language, methods.<Class.method>.{ tool | discriminator{field,present,
 *   absent} | positional[] | return_envelope } }.
 * Node's lowercase namespacing (`resend.emails.send`) is matched case-insensitively
 * against the TitleCase surface keys (`Emails.send`, from the descriptor original_function).
 *
 * Env (mirrors the Python runtime, injected into the skill's Bash subprocess):
 *   PLURAI_CODE_SURFACE — path to the bench code_surface.yaml (required).
 *   PLURAI_SESSION_ID   — active session id (required).
 *   PLURAI_MOCK_URL     — mock service base url (required).
 *   PLURAI_URL / PLURAI_TOKEN — the resolved backend pair the Python side passes in. The bearer
 *     goes only to a mock with exactly PLURAI_URL's origin (bearerFor), never to another loopback port.
 * Fail-closed: a missing env var, an unmapped method, or an unbound non-dict positional
 * throws — never a silent fall-through to the real vendor API.
 */

const fs = require('node:fs');
const { resolve: resolvePath } = require('node:path');
const http = require('node:http');
const https = require('node:https');
const Module = require('node:module');
const { URL } = require('node:url');
const retry = require('./retry.cjs');

class CodeSurfaceError extends Error {}
class CodeInterceptConfigError extends Error {}
// Distinct from CodeInterceptConfigError (env/config wiring): raised when the /mock/execute
// transport itself fails (HTTP error, non-JSON body, timeout, or a mock error envelope), so a
// caller can tell a misconfiguration apart from a runtime mock/transport failure.
class CodeInterceptTransportError extends Error {}

// Schema versions this loader understands; kept in lock-step with the Python
// _SUPPORTED_SURFACE_VERSIONS. A surface declaring any other version fails closed.
//   v1: `modules`-only module-shadow surfaces.
//   v2: adds the top-level `cli_http` section (the CLI/fetch_shadow channel) and makes
//       `modules` optional when `cli_http` is present. v1 surfaces still load unchanged.
const SUPPORTED_SURFACE_VERSIONS = new Set([1, 2]);

// The native return-envelope values a method / cli subcommand may declare; `node_cli_json`
// is the CLI channel's envelope (subprocess emits the mock response as JSON on stdout).
const RETURN_ENVELOPES = new Set(['python_dict', 'node_data_error', 'node_cli_json']);

// The cli_http channel's fixed top-level field values. READ from the authored surface and
// validated for presence + well-formedness here; never derived.
const CLI_HTTP_RUNTIME = 'fetch_shadow';
const CLI_HTTP_RETURN_ENVELOPE = 'node_cli_json';
const CLI_HTTP_UNMAPPED_EGRESS = 'fail_closed';

function requireNonEmptyString(value, what, codeSurfacePath) {
  if (typeof value !== 'string' || value === '') {
    throw new CodeSurfaceError(`${codeSurfacePath}: ${what} must be a non-empty string (got ${JSON.stringify(value)}).`);
  }
  return value;
}

// --- Minimal YAML reader, scoped to the code_surface schema ------------------
// Block-style maps by 2-space indentation, `key: value`, `[a, b]` flow lists, `- item`
// block sequences, `#` comments, and quoted scalars. Deliberately small: it parses ONLY
// the shapes the code_surface emits (it is not a general YAML parser). It reads the same
// schema as the Python loader.
//
// Block sequences matter because the SDK projector emits `positional` via PyYAML's safe_dump,
// whose default list style is a block sequence (`positional:\n  - id`), while a hand-written
// surface may use the flow list (`positional: [id]`) — the loader accepts both.
function parseCodeSurface(text) {
  const root = {};
  // Stack of open containers by indent. A `pending` map key (with an empty value) is held
  // until its first child line decides whether the value is a nested MAP (`key: value`
  // child) or a block SEQUENCE (`- item` child) — a block sequence has the same leading
  // indent as a map child would, so the type can't be known until the child is seen.
  const stack = [{ indent: -1, container: root }];
  let pending = null; // { indent, container, key } — an empty-value key awaiting its child
  const lines = text.split(/\r?\n/);
  for (const rawLine of lines) {
    const line = stripComment(rawLine);
    if (line.trim() === '') continue;
    const indent = line.length - line.trimStart().length;
    const content = line.trim();

    // Block-sequence item: `- value` (or a bare `-`). Belongs to the pending empty-value key;
    // the key's value becomes a list (created on first item). PyYAML's safe_dump puts the
    // dash at the SAME indent as the parent key (`positional:\n- id` after dump, re-indented
    // to the key's column when nested), so accept indent >= the key's indent — not strictly
    // deeper. A bare top-level sequence (no pending key) is not a code_surface shape → throw.
    if (content === '-' || content.startsWith('- ')) {
      if (pending == null || indent < pending.indent) {
        throw new CodeSurfaceError(`code_surface: block-sequence item with no parent key: ${JSON.stringify(rawLine)}`);
      }
      let list = pending.container[pending.key];
      if (!Array.isArray(list)) {
        list = [];
        pending.container[pending.key] = list;
      }
      const itemText = content === '-' ? '' : content.slice(2).trim();
      list.push(parseScalar(itemText));
      continue;
    }

    const colon = content.indexOf(':');
    if (colon === -1) {
      throw new CodeSurfaceError(`code_surface: cannot parse line ${JSON.stringify(rawLine)}`);
    }
    const key = content.slice(0, colon).trim();
    const valueText = content.slice(colon + 1).trim();

    // A `key:` child of a pending empty-value key means that value is a nested MAP — open it
    // as a real container frame before placing this child. (If instead a `- item` had come,
    // the block above would have made the value a list and this never runs.)
    if (pending != null && indent > pending.indent) {
      const child = {};
      pending.container[pending.key] = child;
      stack.push({ indent: pending.indent, container: child });
      pending = null;
    }

    while (stack.length > 1 && indent <= stack[stack.length - 1].indent) {
      stack.pop();
    }
    pending = null; // a sibling/dedent line ends any still-open pending key
    const parent = stack[stack.length - 1].container;
    if (valueText === '') {
      // Empty value — defer: a nested map or a block sequence, decided by the first child.
      parent[key] = {};
      pending = { indent, container: parent, key };
    } else {
      parent[key] = parseScalar(valueText);
    }
  }
  return root;
}

function stripComment(line) {
  // Strip a `#` comment that is not inside quotes. The surface has no `#` inside scalars,
  // but guard quotes anyway so a future value containing `#` is safe.
  let inSingle = false;
  let inDouble = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === "'" && !inDouble) inSingle = !inSingle;
    else if (ch === '"' && !inSingle) inDouble = !inDouble;
    else if (ch === '#' && !inSingle && !inDouble) return line.slice(0, i);
  }
  return line;
}

function parseScalar(text) {
  if (text.startsWith('[') && text.endsWith(']')) {
    const inner = text.slice(1, -1).trim();
    if (inner === '') return [];
    return inner.split(',').map((item) => parseScalar(item.trim()));
  }
  if (
    (text.startsWith('"') && text.endsWith('"')) ||
    (text.startsWith("'") && text.endsWith("'"))
  ) {
    return text.slice(1, -1);
  }
  if (/^-?\d+$/.test(text)) return parseInt(text, 10);
  if (text === 'true') return true;
  if (text === 'false') return false;
  if (text === 'null' || text === '~') return null;
  return text;
}

// --- Surface model -----------------------------------------------------------

function loadSurfaceDocument(codeSurfacePath) {
  // Parse + STRUCTURALLY validate a code_surface.yaml into { modules, cliHttp } — the Node
  // analog of the Python _load_surface_document, kept in lock-step. The validation ORDER is
  // load-bearing: version FIRST (a cli_http-only v2 surface has no modules, so a presence
  // check run first would reject it), THEN the per-section presence check (at least one of modules /
  // cli_http), THEN per-section structural validation. READS the authored surface — it
  // never derives the mapping.
  const raw = parseCodeSurface(fs.readFileSync(codeSurfacePath, 'utf8'));
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new CodeSurfaceError(`${codeSurfacePath}: surface must be a mapping.`);
  }

  // Version FIRST, so a per-section presence check cannot shadow the version dispatch. A
  // cli_http-only v2 surface has no `modules`, so an unconditional modules-required guard
  // would throw before we got here. The
  // projector always stamps `version`; an absent version is anomalous and raising is safer
  // than defaulting to any version (lock-step with the Python loader).
  if (raw.version == null) {
    throw new CodeSurfaceError(
      `${codeSurfacePath}: surface is missing the required 'version' field. ` +
        'The SDK projector always stamps this; an absent version is anomalous.',
    );
  }
  const version = raw.version;
  if (!SUPPORTED_SURFACE_VERSIONS.has(version)) {
    const supported = [...SUPPORTED_SURFACE_VERSIONS].sort((a, b) => a - b).join(', ');
    throw new CodeSurfaceError(
      `${codeSurfacePath}: unsupported code_surface schema version ${JSON.stringify(version)} ` +
        `(this loader understands version(s) ${supported}). Refusing to guess at an unknown schema.`,
    );
  }

  const hasModules = raw.modules != null;
  const hasCliHttp = raw.cli_http != null;
  if (!hasModules && !hasCliHttp) {
    throw new CodeSurfaceError(
      `${codeSurfacePath}: surface must declare at least one of top-level 'modules' ` +
        `(SDK module-shadow channel) or 'cli_http' (CLI/fetch_shadow channel).`,
    );
  }

  const modules = hasModules ? parseModules(raw.modules, codeSurfacePath) : {};
  const cliHttp = hasCliHttp ? parseCliHttp(raw.cli_http, codeSurfacePath) : null;
  return { modules, cliHttp };
}

function parseModules(rawModules, codeSurfacePath) {
  // Parse + validate the `modules` section into { moduleName: { language, methods, byLowerKey } }.
  if (!rawModules || typeof rawModules !== 'object' || Array.isArray(rawModules)) {
    throw new CodeSurfaceError(`${codeSurfacePath}: top-level 'modules' must be a mapping.`);
  }
  // Known language values — kept in lock-step with the Python loader's known set.
  const KNOWN_LANGUAGES = new Set(['python', 'node']);

  const surfaces = {};
  for (const [moduleName, moduleSpec] of Object.entries(rawModules)) {
    const language = moduleSpec.language || 'python';
    if (!KNOWN_LANGUAGES.has(language)) {
      throw new CodeSurfaceError(
        `${codeSurfacePath}: module '${moduleName}' declares unknown language ` +
          `${JSON.stringify(language)} (known: ${[...KNOWN_LANGUAGES].join(', ')}).`,
      );
    }
    const methods = {};
    const byLowerKey = {};
    for (const [methodPath, methodSpec] of Object.entries(moduleSpec.methods || {})) {
      const where = `${moduleName}.${methodPath}`;
      let tool = methodSpec.tool;
      const disc = methodSpec.discriminator;
      if ((tool == null) === (disc == null)) {
        throw new CodeSurfaceError(
          `${codeSurfacePath}: method ${where} must set exactly one of 'tool' or 'discriminator'.`,
        );
      }
      if (tool != null) {
        tool = requireNonEmptyString(tool, `method ${where} 'tool'`, codeSurfacePath);
      }
      let discriminator = null;
      if (disc != null) {
        if (typeof disc !== 'object' || Array.isArray(disc)) {
          throw new CodeSurfaceError(`${codeSurfacePath}: discriminator for ${where} must be a mapping.`);
        }
        for (const field of ['field', 'present', 'absent']) {
          if (!(field in disc)) {
            throw new CodeSurfaceError(`${codeSurfacePath}: discriminator for ${where} is missing '${field}'.`);
          }
        }
        discriminator = {
          field: requireNonEmptyString(disc.field, `discriminator 'field' for ${where}`, codeSurfacePath),
          present: requireNonEmptyString(disc.present, `discriminator 'present' for ${where}`, codeSurfacePath),
          absent: requireNonEmptyString(disc.absent, `discriminator 'absent' for ${where}`, codeSurfacePath),
        };
      }
      const positional = methodSpec.positional || [];
      if (!Array.isArray(positional) || !positional.every((n) => typeof n === 'string')) {
        throw new CodeSurfaceError(
          `${codeSurfacePath}: 'positional' for ${where} must be a list of arg-name strings.`,
        );
      }
      const returnEnvelope = methodSpec.return_envelope || 'python_dict';
      if (!RETURN_ENVELOPES.has(returnEnvelope)) {
        const allowed = [...RETURN_ENVELOPES].join(', ');
        throw new CodeSurfaceError(
          `${codeSurfacePath}: 'return_envelope' for ${where} must be one of {${allowed}} (got ${JSON.stringify(returnEnvelope)}).`,
        );
      }
      const rule = {
        tool: tool ?? null,
        discriminator,
        positional,
        returnEnvelope,
      };
      methods[methodPath] = rule;
      const lowerKey = methodPath.toLowerCase();
      if (Object.prototype.hasOwnProperty.call(byLowerKey, lowerKey)) {
        // Two surface keys that differ only in case would silently collide in the
        // case-insensitive lookup table — detect and raise rather than drop one silently.
        const existing = Object.keys(methods).find((k) => k !== methodPath && k.toLowerCase() === lowerKey);
        throw new CodeSurfaceError(
          `${codeSurfacePath}: method keys '${existing ?? lowerKey}' and '${methodPath}' in module '${moduleName}' ` +
            'collide under case-insensitive matching. Method keys must be unique case-insensitively.',
        );
      }
      byLowerKey[lowerKey] = rule;
    }
    surfaces[moduleName] = { language, methods, byLowerKey };
  }
  return surfaces;
}

function parseCliHttp(rawCliHttp, codeSurfacePath) {
  // Parse + STRUCTURALLY validate the `cli_http` section (Node analog of the Python
  // _parse_cli_http). Validates the fixed channel knobs (runtime == fetch_shadow, preload a
  // non-empty path, return_envelope == node_cli_json, unmapped_egress == fail_closed) and
  // each subcommand (a non-empty `tool` and an `egress_guard` mapping with non-empty
  // `method` + `url_pattern`). STRUCTURAL only: it does NOT check that a discriminator field
  // actually distinguishes two tools per their args — that needs config args and lives at
  // GENERATION time, not the runtime loader. READS the authored surface; derives nothing.
  if (!rawCliHttp || typeof rawCliHttp !== 'object' || Array.isArray(rawCliHttp)) {
    throw new CodeSurfaceError(`${codeSurfacePath}: top-level 'cli_http' must be a mapping.`);
  }

  const runtime = requireNonEmptyString(rawCliHttp.runtime, `'cli_http.runtime'`, codeSurfacePath);
  if (runtime !== CLI_HTTP_RUNTIME) {
    throw new CodeSurfaceError(`${codeSurfacePath}: 'cli_http.runtime' must be ${JSON.stringify(CLI_HTTP_RUNTIME)} (got ${JSON.stringify(runtime)}).`);
  }
  const preload = requireNonEmptyString(rawCliHttp.preload, `'cli_http.preload'`, codeSurfacePath);
  const returnEnvelope = requireNonEmptyString(rawCliHttp.return_envelope, `'cli_http.return_envelope'`, codeSurfacePath);
  if (returnEnvelope !== CLI_HTTP_RETURN_ENVELOPE) {
    throw new CodeSurfaceError(
      `${codeSurfacePath}: 'cli_http.return_envelope' must be ${JSON.stringify(CLI_HTTP_RETURN_ENVELOPE)} (got ${JSON.stringify(returnEnvelope)}).`,
    );
  }
  const unmappedEgress = requireNonEmptyString(rawCliHttp.unmapped_egress, `'cli_http.unmapped_egress'`, codeSurfacePath);
  if (unmappedEgress !== CLI_HTTP_UNMAPPED_EGRESS) {
    throw new CodeSurfaceError(
      `${codeSurfacePath}: 'cli_http.unmapped_egress' must be ${JSON.stringify(CLI_HTTP_UNMAPPED_EGRESS)} (got ${JSON.stringify(unmappedEgress)}).`,
    );
  }

  const rawSubcommands = rawCliHttp.subcommands;
  if (!rawSubcommands || typeof rawSubcommands !== 'object' || Array.isArray(rawSubcommands) || Object.keys(rawSubcommands).length === 0) {
    throw new CodeSurfaceError(
      `${codeSurfacePath}: 'cli_http.subcommands' must be a non-empty mapping of subcommand -> {tool, egress_guard}.`,
    );
  }
  const subcommands = {};
  for (const [name, spec] of Object.entries(rawSubcommands)) {
    const where = `cli_http.subcommands.${name}`;
    if (!spec || typeof spec !== 'object' || Array.isArray(spec)) {
      throw new CodeSurfaceError(`${codeSurfacePath}: ${where} must be a mapping.`);
    }
    const tool = requireNonEmptyString(spec.tool, `${where} 'tool'`, codeSurfacePath);
    const guard = spec.egress_guard;
    if (!guard || typeof guard !== 'object' || Array.isArray(guard)) {
      throw new CodeSurfaceError(`${codeSurfacePath}: ${where} 'egress_guard' must be a mapping.`);
    }
    subcommands[name] = {
      tool,
      egressGuard: {
        method: requireNonEmptyString(guard.method, `${where} 'egress_guard.method'`, codeSurfacePath),
        urlPattern: requireNonEmptyString(guard.url_pattern, `${where} 'egress_guard.url_pattern'`, codeSurfacePath),
      },
    };
  }

  return { runtime, preload, returnEnvelope, unmappedEgress, subcommands };
}

function loadSurfaces(codeSurfacePath) {
  // Backward-compatible accessor for callers that only need the SDK module-shadow surfaces
  // (the import shadow). Delegates to loadSurfaceDocument (so the full document — including
  // cli_http — is validated) and returns just its `modules` map.
  return loadSurfaceDocument(codeSurfacePath).modules;
}

function isNamespacePrefix(surface, path) {
  // True if `path` is a strict dotted prefix of some known method key (case-insensitive,
  // mirroring resolution). The empty root is always a prefix.
  if (path === '') return true;
  const prefix = `${path.toLowerCase()}.`;
  return Object.keys(surface.byLowerKey).some((m) => m.startsWith(prefix));
}

function isKnownAttribute(surface, parentPath, name) {
  // Node analog of the Python _ModuleSurface.is_known_attribute: allow walking into a
  // modeled namespace or onto a method, plus an unmapped LEAF under a NAMED namespace
  // (fails at call); reject an unmodeled top-level attribute (e.g. `resend.errors`).
  const childLower = (parentPath ? `${parentPath}.${name}` : name).toLowerCase();
  if (childLower in surface.byLowerKey) return true;
  if (isNamespacePrefix(surface, parentPath ? `${parentPath}.${name}` : name)) return true;
  return Boolean(parentPath) && isNamespacePrefix(surface, parentPath);
}

function resolveTool(rule, args) {
  if (rule.tool != null) return rule.tool;
  const field = rule.discriminator.field;
  return field in args ? rule.discriminator.present : rule.discriminator.absent;
}

function mergeCallArgs(rule, methodPath, callArgs) {
  // Mirror of the Python _merge_call_args: bind rule-declared positionals by index, merge
  // an unbound object positional (the `send(params, options)` shape), and FAIL CLOSED on
  // an unbound non-object positional (a bare id with no binding slot). Too FEW positionals
  // (fewer args than declared names) also fails closed — Node has no kwargs, so every
  // declared positional name must be supplied by position.
  if (callArgs.length < rule.positional.length) {
    throw new CodeSurfaceError(
      `'${methodPath}' declares ${rule.positional.length} positional arg(s) ` +
        `${JSON.stringify(rule.positional)} but was called with ${callArgs.length}. ` +
        'Refusing to dispatch a partial arg map.',
    );
  }
  const merged = {};
  callArgs.forEach((arg, i) => {
    if (i < rule.positional.length) {
      merged[rule.positional[i]] = arg;
    } else if (arg && typeof arg === 'object' && !Array.isArray(arg)) {
      Object.assign(merged, arg);
    } else {
      throw new CodeSurfaceError(
        `positional arg ${i} (${JSON.stringify(arg)}) to '${methodPath}' has no binding: the method ` +
          `declares ${rule.positional.length} positional name(s) ${JSON.stringify(rule.positional)} and the ` +
          'arg is not a params object. Add it to the method `positional` list in the code_surface.',
      );
    }
  });
  return merged;
}

function shapeNativeReturn(response, returnEnvelope) {
  // Node-native return: resend-node yields `{ data, error }`. `python_dict` (the resend
  // surface default) would hand back the bare dict, but a Node skill always destructures
  // `{ data, error }`, so node_data_error is the shape the skill expects.
  if (returnEnvelope === 'node_data_error') {
    return { data: response, error: null };
  }
  return response;
}

// --- Frozen-wire dispatch (POST /mock/execute) -------------------------------

function readEnvConfig() {
  const sid = process.env.PLURAI_SESSION_ID;
  const mockUrl = process.env.PLURAI_MOCK_URL;
  const missing = [];
  if (!sid) missing.push('PLURAI_SESSION_ID');
  if (!mockUrl) missing.push('PLURAI_MOCK_URL');
  if (missing.length) {
    throw new CodeInterceptConfigError(
      `code action-channel interception is not wired: missing ${missing.join(', ')} in the environment. ` +
        'Refusing to reach the real vendor API.',
    );
  }
  return { sid, mockUrl };
}

function currentTurn() {
  // The OUTER conversation turn each /mock/execute event is filed under. This process is a
  // nested session spawned by the child that drives the turn, so the index arrives in the
  // env (PLURAI_TURN_INDEX, written by plurai._mocking.turn_context). Absent or malformed
  // means "nobody reported a turn" — dispatch on 0, never fail a tool call over a
  // breadcrumb. Read per call so a re-exec'd process picks up the current value.
  const raw = process.env.PLURAI_TURN_INDEX;
  if (raw === undefined) return 0;
  const parsed = Number.parseInt(String(raw).trim(), 10);
  return Number.isNaN(parsed) ? 0 : parsed;
}

function mockExecuteUrl(base) {
  // The frozen-wire endpoint, resolved RELATIVE to the mock base url.
  //
  // `new URL('/mock/execute', base)` looks equivalent and is not: an absolute path resolves
  // against the ORIGIN and throws the base's path away. A base can carry a path —
  // `http://host/mock/<env_id>/` — and resolving would send every tool call to `http://host/mock/
  // execute`, where the proxy reads `execute` as an env id and answers 410. A mock at its own
  // origin is unaffected.
  //
  // Built by concatenation rather than by URL resolution because that is exactly what the Python
  // channel does (`MockServiceClient` strips trailing slashes and appends the path). The two
  // channels talk to the same wire and must not disagree about where it is.
  return new URL(`${String(base).replace(/\/+$/, '')}/mock/execute`);
}

// The /mock/execute ceiling, from the ONE knob the Python channel reads too
// (plurai._mocking.client.mock_execute_timeout): PLURAI_MOCK_EXECUTE_TIMEOUT in seconds, a finite
// number above zero and at most MAX_MOCK_EXECUTE_TIMEOUT_S, else the default — named once per bad
// raw value on stderr, never fatal. The default and the maximum are the Python constants'
// values; a parity test holds the two languages to them.
const MOCK_EXECUTE_TIMEOUT_ENV = 'PLURAI_MOCK_EXECUTE_TIMEOUT';
const DEFAULT_MOCK_EXECUTE_TIMEOUT_S = 300;
const MAX_MOCK_EXECUTE_TIMEOUT_S = 3600;
const warnedExecuteTimeouts = new Set();

function mockExecuteTimeoutMs() {
  const raw = String(process.env[MOCK_EXECUTE_TIMEOUT_ENV] ?? '').trim();
  if (!raw) return DEFAULT_MOCK_EXECUTE_TIMEOUT_S * 1000;
  const value = Number(raw);
  if (Number.isFinite(value) && value > 0 && value <= MAX_MOCK_EXECUTE_TIMEOUT_S) return value * 1000;
  if (!warnedExecuteTimeouts.has(raw)) {
    warnedExecuteTimeouts.add(raw);
    process.stderr.write(
      `  [plurai] WARNING ${MOCK_EXECUTE_TIMEOUT_ENV}=${JSON.stringify(raw)} is not a finite positive number of seconds ` +
        `up to ${MAX_MOCK_EXECUTE_TIMEOUT_S}; /mock/execute uses the default ${DEFAULT_MOCK_EXECUTE_TIMEOUT_S}s.\n`,
    );
  }
  return DEFAULT_MOCK_EXECUTE_TIMEOUT_S * 1000;
}

function transportError(tool, unavailable) {
  // The failure of one tool call's wire. When the retry budget was spent on an outage (or the backend said it is unavailable)
  // it carries `unavailable` ({ cause }). The process is spawned by the agent and has no channel to the run, so the agent reads
  // the tool error and the run does not stop on it.
  const err = new CodeInterceptTransportError(unansweredToolMessage(tool));
  if (unavailable) err.unavailable = unavailable;
  return err;
}

function unansweredToolMessage(tool) {
  // What the AGENT is shown when /mock/execute did not answer: the same neutral text the semantic
  // channel's stub carries (plurai._mocking.resolution). A non-2xx there is a hop in front of the mock,
  // and its body names harness remedies (a token, deploy_env) a judge would read as the tool's answer.
  return `tool ${JSON.stringify(String(tool))} returned no result (mock transport error)`;
}

// Connections outlive one tool call: a run makes hundreds of small requests to one host, and a new
// handshake for each is most of their cost. Idle sockets are unref'd by the agent, so they never keep
// the process alive.
const keepAliveAgents = { 'http:': new http.Agent({ keepAlive: true }), 'https:': new https.Agent({ keepAlive: true }) };

// The wait between throttled retries goes through `timing.sleep`, which tests replace.
const timing = { sleep: (ms) => new Promise((resolveSleep) => setTimeout(resolveSleep, ms)) };

function postOnce(url, body, headers, timeoutMs, agent = keepAliveAgents[url.protocol]) {
  // One POST, resolving { statusCode, headers, text } for ANY status and rejecting on a socket
  // failure or the timeout. Classifying the answer is the caller's. A timeout that fires before the
  // socket connected rejects with `connectTimeout: true`: the request never left, so repeating it is safe.
  const transport = url.protocol === 'https:' ? https : http;
  return new Promise((resolve, reject) => {
    let connected = false;
    const req = transport.request(
      url,
      { method: 'POST', headers, timeout: timeoutMs, agent },
      (res) => {
        const chunks = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => resolve({ statusCode: res.statusCode ?? 200, headers: res.headers, text: Buffer.concat(chunks).toString('utf8') }));
        res.on('error', reject);
      },
    );
    req.on('error', reject);
    req.on('socket', (socket) => {
      if (socket.connecting) socket.once('connect', () => { connected = true; });
      else connected = true; // a kept-alive socket is already connected
    });
    req.on('timeout', () => req.destroy(Object.assign(new Error('request timed out'), { connectTimeout: !connected })));
    req.write(body);
    req.end();
  });
}

async function postMockWire(url, body, headers, timeoutMs, agent) {
  // POST that waits out what clears by itself, under the verdicts of retry.cjs. A 429 is a rate limiter refusing before it
  // forwards, so the request was never applied and repeating it is safe whatever the tool: the wait follows Retry-After with
  // jitter and a cap, and a throttle that outlasts its budget is returned as the last answer, for the caller to treat as the
  // transport fault it is. A 503, a 502 the proxy marked as never connected to the mock, and a connection that failed while
  // connecting get the transient budget (retry.TRANSIENT_RETRIES retries, about 60 seconds, so a backend upgrade passes
  // unseen); any other 502, a 504 and a connection that broke after the request went out may follow a tool call that already
  // ran, so they are never re-sent. A spent budget, and the backend's typed unavailable refusal, end as the failure or the
  // answer carrying `unavailable` ({ cause }).
  const started = Date.now();
  const backoffWindowMs = retryWindowMs();
  let throttles = 0;
  let outages = 0;
  for (;;) {
    let reply = null;
    let failure = null;
    try {
      reply = await postOnce(url, body, headers, timeoutMs, agent);
    } catch (e) {
      failure = e;
    }
    const verdict = failure !== null ? retry.failureVerdict(failure) : retry.replyVerdict(reply.statusCode, reply.text);
    if (verdict === 'outage') {
      reply.unavailable = { cause: 'backend' };
      return reply;
    }
    if (verdict === 'throttle') {
      if (throttles >= retry.RATE_LIMIT_RETRIES) {
        process.stderr.write(
          `  [plurai] the Plurai backend is still rate limiting this run (HTTP 429) after ${retry.RATE_LIMIT_RETRIES} retries; giving up on ${url.pathname}.\n`,
        );
        return reply;
      }
      throttles += 1;
      const waitMs = retry.rateLimitWaitMs(throttles, reply.headers['retry-after']);
      process.stderr.write(
        `  [plurai] mock throttled (HTTP 429) on ${url.pathname}; retrying in ${(waitMs / 1000).toFixed(1)}s (retry ${throttles}/${retry.RATE_LIMIT_RETRIES})\n`,
      );
      await timing.sleep(waitMs);
      continue;
    }
    if (verdict !== 'wait') {
      if (failure !== null) throw failure;
      return reply;
    }
    outages += 1;
    const delayMs = retry.backoffSeconds(outages) * 1000;
    if (outages > retry.TRANSIENT_RETRIES || Date.now() - started + delayMs > timeoutMs + backoffWindowMs) {
      if (failure !== null) {
        failure.unavailable = { cause: 'transport' };
        throw failure;
      }
      reply.unavailable = { cause: 'status' };
      return reply;
    }
    await timing.sleep(delayMs);
  }
}

function retryWindowMs() {
  // The longest the transient waits may add to one attempt's ceiling: the sum of the back-off waits, as the Python client bounds it.
  let total = 0;
  for (let n = 1; n <= retry.TRANSIENT_RETRIES; n += 1) total += retry.backoffSeconds(n, retry.BACKOFF_BASE_SECONDS, retry.BACKOFF_CAP_SECONDS, () => 0);
  return total * 1000;
}

function isLoopbackHost(hostname) {
  return hostname === 'localhost' || /^127(\.\d{1,3}){3}$/.test(hostname) || hostname === '[::1]';
}

function bearerFor(target) {
  // The token a request to `target` carries: the resolved PLURAI_TOKEN, only when `target` has
  // exactly the origin PLURAI_URL names (scheme, host and port). Anything else, an unparseable
  // url included, gets none (mirrors plurai._endpoints.Endpoint.bearer_for). Loopback is no
  // exception: any local process can listen on a loopback port and would collect a remote
  // backend's token. A local mock needs no token.
  const token = (process.env.PLURAI_TOKEN || '').trim();
  if (!token) return null;
  let wanted;
  try {
    wanted = new URL(target);
  } catch (_e) {
    return null;
  }
  try {
    const backend = new URL(process.env.PLURAI_URL);
    // Plain http carries the token in clear text: only a loopback backend may use it.
    if (backend.protocol !== 'https:' && !isLoopbackHost(backend.hostname)) return null;
    return backend.origin === wanted.origin ? token : null;
  } catch (_e) {
    return null;
  }
}

const CLIENT_HEADER = 'X-Plurai-Client';
const CLIENT_NAME = 'plurai-simulation-sdk';

function sdkVersion() {
  // The installed SDK version, read off the sibling `plurai_simulation_sdk-<version>.dist-info` folder of the
  // package this file ships in (the folder name carries the version, local segment included). A source checkout
  // has none beside it and reports 0.0.0, the version the Python side reports without package metadata.
  const siteDir = resolvePath(__dirname, '..', '..', '..');
  try {
    for (const entry of fs.readdirSync(siteDir)) {
      const found = /^plurai_simulation_sdk-([^-]+)\.dist-info$/.exec(entry);
      if (found) return found[1].split('+')[0];
    }
  } catch (_e) {
    // an unreadable site folder is the same absence
  }
  return '0.0.0';
}

const CLIENT_HEADER_VALUE = `${CLIENT_NAME}/${sdkVersion()}`;

function mockExecuteRequest(tool, args, sid, mockUrl) {
  // The wire request for one tool call: the url, the JSON body and the headers.
  const body = JSON.stringify({ session_id: sid, tool, args, turn: currentTurn() });
  const headers = { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body), [CLIENT_HEADER]: CLIENT_HEADER_VALUE };
  const url = mockExecuteUrl(mockUrl);
  const token = bearerFor(url.href);
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return { url, body, headers };
}

async function postMockExecute(tool, args, sid, mockUrl) {
  // POST /mock/execute for one tool call; resolves { status, body } with the inner `response`.
  // Rejects with CodeInterceptTransportError when the wire itself fails: a socket error or timeout,
  // a non-2xx status (the mock never answers a tool call with one, so it is a hop in front of the
  // mock, the tool never ran, and its body is not the tool's answer), a non-JSON body.
  const { url, body, headers } = mockExecuteRequest(tool, args, sid, mockUrl);
  let reply;
  try {
    reply = await postMockWire(url, body, headers, mockExecuteTimeoutMs());
  } catch (e) {
    throw transportError(tool, e.unavailable);
  }
  if (reply.statusCode < 200 || reply.statusCode >= 300) {
    throw transportError(tool, reply.unavailable);
  }
  let parsed;
  try {
    parsed = JSON.parse(reply.text);
  } catch (_e) {
    throw transportError(tool, undefined);
  }
  // {response, state_version} on success; {error, status} on a bad session etc.
  // An error envelope (no `response` key) must REJECT — wrapping it as a success
  // value would let `{ data: <error>, error: null }` reach the Node skill (which
  // destructures `{ data, error }`) and look like a successful send. The Python
  // runtime surfaces the envelope to its own error handling; the Node skill has no
  // such seam here, so we reject so the awaited call throws.
  if (!parsed || typeof parsed !== 'object' || !('response' in parsed)) {
    throw new CodeInterceptTransportError(`/mock/execute returned an error envelope: ${reply.text}`);
  }
  return { status: reply.statusCode, body: parsed.response };
}

async function executeMock(tool, args) {
  // Resolves with the inner `response` (resend-node methods are async, so the shadow methods return
  // Promises), mirroring the Python code_intercept_call.
  const { sid, mockUrl } = readEnvConfig();
  return (await postMockExecute(tool, args, sid, mockUrl)).body;
}

// --- The shadow module: `new Resend(key)` → namespace proxies ----------------

function buildShadowModule(surface, dispatch) {
  // resend-node shape: `import { Resend } from 'resend'; const r = new Resend(key)`.
  // The instance exposes lowercase namespaces (r.emails, r.templates); each method call
  // r.<ns>.<method>(args) resolves `<Ns>.<method>` case-insensitively against the surface.
  function makeNamespaceProxy(namespace) {
    return new Proxy(
      {},
      {
        get(_target, method) {
          if (typeof method !== 'string') return undefined;
          const methodPath = `${namespace}.${method}`;
          const rule = surface.byLowerKey[methodPath.toLowerCase()];
          return async (...callArgs) => {
            if (rule == null) {
              throw new CodeSurfaceError(
                `no code_surface mapping for method path '${methodPath}' ` +
                  `(known: ${Object.keys(surface.methods).join(', ')}).`,
              );
            }
            const args = mergeCallArgs(rule, methodPath, callArgs);
            const tool = resolveTool(rule, args);
            const response = await dispatch(tool, args);
            return shapeNativeReturn(response, rule.returnEnvelope);
          };
        },
      },
    );
  }

  class Resend {
    constructor() {
      return new Proxy(this, {
        get(_target, namespace) {
          if (typeof namespace !== 'string') return undefined;
          // Fail-closed attribute access (Node analog of the Python proxy guard): only
          // walk into a modeled namespace. An unmodeled top-level attribute (e.g.
          // `resend.errors`) returns undefined rather than a proxy that masquerades as a
          // real value — Node has no `except`, but the same masking risk applies to a
          // skill probing `resend.someUnknownThing`.
          if (!isKnownAttribute(surface, '', namespace)) return undefined;
          return makeNamespaceProxy(namespace);
        },
      });
    }
  }

  const exported = { Resend, __plurai_code_shadow__: true };
  exported.default = exported;
  return exported;
}

// --- Install: patch CJS require + register an ESM loader ----------------------

function install(codeSurfacePath, dispatch = executeMock) {
  const surfaces = loadSurfaces(codeSurfacePath);
  const shadows = {};
  for (const [moduleName, surface] of Object.entries(surfaces)) {
    shadows[moduleName] = buildShadowModule(surface, dispatch);
  }
  const shadowNames = Object.keys(shadows);

  // CommonJS: intercept require('resend'). Patch Module._load so a cached real module is
  // bypassed and the shadow is returned for the named vendor modules only.
  const originalLoad = Module._load;
  Module._load = function patchedLoad(request, parent, isMain) {
    if (Object.prototype.hasOwnProperty.call(shadows, request)) {
      return shadows[request];
    }
    return originalLoad.apply(this, arguments);
  };

  // ESM: register a loader hook so `import 'resend'` is shadowed too (Node >= 20.6). The
  // resend skill examples are ESM, so this is the path that actually fires for them; the
  // CJS patch above covers a require()-based snippet.
  //
  // Fail-closed: if the surface names ≥1 module to shadow but `module.register` is
  // unavailable (Node < 20.6), the CJS patch alone does NOT intercept `import { Resend }
  // from 'resend'` — the real package would load and a real send could escape. Refuse to
  // run rather than leave an un-shadowed ESM import reachable.
  //
  // ESM is WIRE-ONLY: the loader sidecar runs on a separate module thread and the
  // source it returns rebuilds the shadow on the main thread bound to `executeMock`. A
  // function `dispatch` cannot cross that boundary via `module.register` `data`, so a
  // custom (test) dispatch is honored for CJS/`buildShadowModule` only. Production always
  // uses the wire default, so this is not a runtime gap; we assert it explicitly so a
  // caller passing a custom dispatch and expecting it on the ESM path fails loudly.
  if (shadowNames.length > 0) {
    if (typeof Module.register !== 'function') {
      Module._load = originalLoad; // undo the CJS patch — we are aborting the install
      throw new CodeInterceptConfigError(
        `code action-channel ESM shadowing requires Node >= 20.6 (module.register), but it is ` +
          `unavailable; with ${shadowNames.length} module(s) to shadow (${shadowNames.join(', ')}), an ` +
          `\`import 'resend'\` would reach the REAL vendor package. Refusing to run on this Node version.`,
      );
    }
    if (dispatch !== executeMock) {
      throw new CodeInterceptConfigError(
        'a custom dispatch was supplied to install(): the ESM loader path is wire-only (it always ' +
          'uses executeMock and cannot receive a function across the module-thread boundary), so the ' +
          'custom dispatch would NOT apply to `import` statements. Use buildShadowModule(surface, dispatch) ' +
          'directly for non-wire dispatch (the test seam), or install() with the default wire dispatch.',
      );
    }
    const loaderData = { shadowNames };
    let loaderUrl;
    try {
      loaderUrl = new URL('./code_hook_loader.mjs', require('node:url').pathToFileURL(__filename));
    } catch (e) {
      throw new CodeInterceptConfigError(`cannot locate the ESM loader sidecar next to ${__filename}: ${e.message}`);
    }
    Module.register(loaderUrl, { parentURL: require('node:url').pathToFileURL(__filename), data: loaderData });
  }

  // Expose for the ESM loader (runs in a worker thread) and for tests.
  globalThis.__plurai_code_shadows__ = shadows;
  return { uninstall: () => (Module._load = originalLoad), shadows };
}

// --- Entry: --require runs this top-level -----------------------------------

if (process.env.PLURAI_CODE_SURFACE) {
  install(process.env.PLURAI_CODE_SURFACE);
}

module.exports = {
  install,
  loadSurfaces,
  loadSurfaceDocument,
  parseModules,
  parseCliHttp,
  parseCodeSurface,
  mergeCallArgs,
  resolveTool,
  shapeNativeReturn,
  executeMock,
  postMockExecute,
  postMockWire,
  bearerFor,
  mockExecuteRequest,
  CLIENT_HEADER_VALUE,
  timing,
  mockExecuteUrl,
  mockExecuteTimeoutMs,
  unansweredToolMessage,
  currentTurn,
  buildShadowModule,
  isKnownAttribute,
  CodeSurfaceError,
  CodeInterceptConfigError,
  CodeInterceptTransportError,
};
