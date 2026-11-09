/*
 * ESM loader sidecar for the Node `code` action-channel hook (registered by
 * code_hook.cjs via `module.register`). Customization hooks run on a separate module
 * thread and cannot touch the main thread's `Module._load` patch, so for a shadowed
 * specifier (`resend`) the `load` hook returns SOURCE that — when executed on the main
 * thread — rebuilds the shadow from PLURAI_CODE_SURFACE and re-exports `Resend`. This is
 * what makes `import { Resend } from 'resend'` resolve to the shadow under `--require`/
 * `--import`.
 *
 * The set of shadowed names is passed as `data` at register() time; everything else is
 * driven by the same code_surface.yaml the CJS side loads (single source of truth).
 */

let shadowNames = [];

export async function initialize(data) {
  shadowNames = (data && data.shadowNames) || [];
}

export async function resolve(specifier, context, nextResolve) {
  if (shadowNames.includes(specifier)) {
    // Give the synthetic module a stable URL so `load` recognises it.
    return { url: `plurai-code-shadow:${specifier}`, shortCircuit: true };
  }
  return nextResolve(specifier, context);
}

export async function load(url, context, nextLoad) {
  if (url.startsWith('plurai-code-shadow:')) {
    const moduleName = url.slice('plurai-code-shadow:'.length);
    // Source executes on the MAIN thread: require the CJS hook, rebuild this module's
    // shadow from the env-provided surface, and surface `Resend` + a default export.
    const source = `
      import { createRequire } from 'node:module';
      const require = createRequire(${JSON.stringify(import.meta.url)});
      const hook = require('./code_hook.cjs');
      const surfacePath = process.env.PLURAI_CODE_SURFACE;
      if (!surfacePath) {
        throw new hook.CodeInterceptConfigError(
          'PLURAI_CODE_SURFACE is not set in the ESM-shadowed import of ' + ${JSON.stringify(moduleName)} +
            '; cannot rebuild the code-channel shadow. The skill subprocess must inject the code_surface.yaml path.',
        );
      }
      let surfaces;
      try {
        surfaces = hook.loadSurfaces(surfacePath);
      } catch (e) {
        throw new hook.CodeInterceptConfigError(
          'failed to load PLURAI_CODE_SURFACE (' + surfacePath + ') for the ESM-shadowed import of ' +
            ${JSON.stringify(moduleName)} + ': ' + e.message,
        );
      }
      const surface = surfaces[${JSON.stringify(moduleName)}];
      if (!surface) {
        throw new hook.CodeInterceptConfigError(
          'PLURAI_CODE_SURFACE (' + surfacePath + ') does not define a shadow for ' + ${JSON.stringify(moduleName)} + '.',
        );
      }
      const shadow = hook.buildShadowModule(surface, hook.executeMock);
      export const Resend = shadow.Resend;
      export const __plurai_code_shadow__ = true;
      export default shadow;
    `;
    return { format: 'module', shortCircuit: true, source };
  }
  return nextLoad(url, context);
}
