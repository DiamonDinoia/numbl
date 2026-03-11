/**
 * Pure parsing of mip directives from .m source files.
 * No Node.js dependencies — safe to import in the browser.
 */

export interface MipDirective {
  type: "load";
  packageName: string;
  line: number; // 1-based
}

export interface MipDirectiveResult {
  directives: MipDirective[];
  cleanedSource: string; // directive lines replaced with blank lines to preserve line numbering
}

const MIP_LOAD_RE = /^\s*mip\s+load\s+((?:[^\s;]+\s*)+?)\s*;?\s*(%.*)?$/;
const MIP_ANY_RE = /^\s*mip\s+/;
const BLANK_OR_COMMENT_RE = /^\s*(%.*)?$/;

export function extractMipDirectives(
  source: string,
  filename: string
): MipDirectiveResult {
  const lines = source.split("\n");
  const directives: MipDirective[] = [];
  let codeStarted = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const lineNum = i + 1;

    if (BLANK_OR_COMMENT_RE.test(line)) {
      continue;
    }

    const loadMatch = line.match(MIP_LOAD_RE);
    if (loadMatch) {
      if (codeStarted) {
        throw new Error(
          `${filename}:${lineNum}: mip directive must appear before any code`
        );
      }
      const packageNames = loadMatch[1].trim().split(/\s+/);
      for (const packageName of packageNames) {
        directives.push({
          type: "load",
          packageName,
          line: lineNum,
        });
      }
      lines[i] = "";
      continue;
    }

    if (MIP_ANY_RE.test(line)) {
      throw new Error(`${filename}:${lineNum}: unknown mip directive`);
    }

    codeStarted = true;
  }

  return { directives, cleanedSource: lines.join("\n") };
}
