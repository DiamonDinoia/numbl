/**
 * MIP directive handling for the CLI (Node.js).
 * Re-exports pure parsing from mip-directives-core.ts and adds
 * the Node-specific processMipLoad() implementation.
 */

import { readFileSync, existsSync } from "fs";
import { join, dirname, basename, extname } from "path";
import { homedir } from "os";
import { executeCode } from "./numbl-core/executeCode.js";
import { toString, RTV } from "./numbl-core/runtime/index.js";

// Re-export pure parsing (no Node deps)
export {
  extractMipDirectives,
  type MipDirective,
  type MipDirectiveResult,
} from "./mip-directives-core.js";

export function processMipLoad(packageName: string): string[] {
  const mipDir = process.env.MIP_DIR || join(homedir(), ".mip");
  const pkgDir = join(mipDir, "packages", packageName);
  const loadScript = join(pkgDir, "load_package.m");

  if (!existsSync(loadScript)) {
    throw new Error(
      `mip load ${packageName}: package not found (expected ${loadScript})`
    );
  }

  const source = readFileSync(loadScript, "utf-8");
  const collectedPaths: string[] = [];

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const customBuiltins: Record<string, (nargout: number, args: any[]) => any> =
    {
      addpath: (_nargout, args) => {
        for (const arg of args) {
          const p = toString(arg);
          // addpath supports multiple dirs separated by pathsep
          for (const part of p.split(":")) {
            if (part.length > 0) {
              collectedPaths.push(part);
            }
          }
        }
        return undefined;
      },
      mfilename: (_nargout, args) => {
        if (args.length > 0 && toString(args[0]) === "fullpath") {
          // Return full path without .m extension
          return RTV.char(loadScript.replace(/\.m$/, ""));
        }
        return RTV.char("load_package");
      },
      fileparts: (nargout, args) => {
        const p = toString(args[0]);
        const dir = dirname(p);
        const ext = extname(p);
        const name = basename(p, ext);
        if (nargout <= 1) return RTV.char(dir);
        if (nargout === 2) return [RTV.char(dir), RTV.char(name)];
        return [RTV.char(dir), RTV.char(name), RTV.char(ext)];
      },
      fullfile: (_nargout, args) => {
        const parts = args.map(a => toString(a));
        return RTV.char(join(...parts));
      },
    };

  executeCode(source, { customBuiltins }, [], loadScript);

  return collectedPaths;
}
