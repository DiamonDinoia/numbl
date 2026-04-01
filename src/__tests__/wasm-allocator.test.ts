import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { createRequire } from "node:module";
import { executeCode } from "../numbl-core/executeCode.js";
import {
  isRuntimeTensor,
  type FloatXArrayType,
} from "../numbl-core/runtime/types.js";
import { _setPoolAlloc } from "../numbl-core/runtime/constructors.js";

const addonPath = join(process.cwd(), "build", "Release", "numbl_addon.node");
const addonExists = existsSync(addonPath);

describe.skipIf(!addonExists)("native pool allocator integration", () => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let addon: any;

  beforeAll(() => {
    const req = createRequire(import.meta.url);
    addon = req(addonPath);
    if (typeof addon.poolInit !== "function") {
      throw new Error("addon does not support pool allocator");
    }
    addon.poolInit(true);
    _setPoolAlloc(
      (n: number) => addon.poolAllocFloat64(n),
      (source: FloatXArrayType) => addon.poolAllocFloat64From(source)
    );
  });

  afterAll(() => {
    _setPoolAlloc(null, null);
    if (addon?.poolInit) addon.poolInit(false);
  });

  it("creates a matrix", () => {
    const result = executeCode("A = [1 2; 3 4];");
    const A = result.variableValues["A"];
    expect(isRuntimeTensor(A)).toBe(true);
    if (isRuntimeTensor(A)) {
      expect(A.shape).toEqual([2, 2]);
      expect(Array.from(A.data)).toEqual([1, 3, 2, 4]);
    }
  });

  it("performs element-wise addition", () => {
    const result = executeCode("A = [1 2; 3 4]; B = A + A;");
    const B = result.variableValues["B"];
    expect(isRuntimeTensor(B)).toBe(true);
    if (isRuntimeTensor(B)) {
      expect(Array.from(B.data)).toEqual([2, 6, 4, 8]);
    }
  });

  it("performs matrix multiply", () => {
    const result = executeCode("A = [1 2; 3 4]; C = A * A;");
    const C = result.variableValues["C"];
    expect(isRuntimeTensor(C)).toBe(true);
    if (isRuntimeTensor(C)) {
      expect(C.shape).toEqual([2, 2]);
      expect(Array.from(C.data)).toEqual([7, 15, 10, 22]);
    }
  });

  it("creates a range", () => {
    const result = executeCode("x = 1:100;");
    const x = result.variableValues["x"];
    expect(isRuntimeTensor(x)).toBe(true);
    if (isRuntimeTensor(x)) {
      expect(x.shape).toEqual([1, 100]);
      expect(x.data[0]).toBe(1);
      expect(x.data[99]).toBe(100);
    }
  });

  it("indexes into a range", () => {
    const result = executeCode("x = 1:100; y = x(1:10);");
    const y = result.variableValues["y"];
    expect(isRuntimeTensor(y)).toBe(true);
    if (isRuntimeTensor(y)) {
      expect(Array.from(y.data)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }
  });

  it("pool is enabled", () => {
    expect(addon.poolEnabled()).toBe(true);
  });
});
