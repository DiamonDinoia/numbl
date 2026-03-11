/**
 * JIT compiler: bridges the runtime with the lowering/codegen layers.
 *
 * Encapsulates resolution caching, compilation caching, and on-demand
 * code generation for functions discovered at runtime.
 */

import { LoweringContext } from "../lowering/loweringContext.js";
import type {
  FileASTCache,
  FunctionIndex,
} from "../lowering/loweringContext.js";
import { Codegen } from "../codegen/codegen.js";
import { resolveFunction, type ResolvedTarget } from "../functionResolve.js";
import { typeToString, type ItemType } from "../lowering/itemTypes.js";
import type { CallSite } from "../runtime/runtimeHelpers.js";
import type { Runtime } from "../runtime/runtime.js";

export class JitCompiler {
  // ── Caches ──────────────────────────────────────────────────────────
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private compileCacheMap = new Map<string, ((...args: any[]) => any) | null>();
  private resolveCacheMap = new Map<string, ResolvedTarget | null>();
  private jitInProgress = new Set<string>();

  private onJitCompile?: (description: string, jsCode: string) => void;

  constructor(
    private ctx: LoweringContext,
    private fileSources: Map<string, string>,
    private functionIndex: FunctionIndex,
    private fileASTCache: FileASTCache,
    private rt: Runtime,
    onJitCompile?: (description: string, jsCode: string) => void,
    private noLineTracking?: boolean
  ) {
    this.onJitCompile = onJitCompile;
  }

  // ── Public: wire up runtime callbacks ────────────────────────────────

  install(): void {
    this.rt.compileSpecialized = (name, argTypes, callSite) =>
      this.compileSpecialized(name, argTypes, callSite);
    this.rt.resolveClassMethod = (className, methodName) =>
      this.resolveClassMethod(className, methodName);
    this.rt.getClassParent = className =>
      this.ctx.getClassInfo(className)?.superClass ?? null;
  }

  // ── Resolution ──────────────────────────────────────────────────────

  private resolve(
    name: string,
    argTypes: ItemType[],
    callSite: CallSite
  ): ResolvedTarget | null {
    const key = JSON.stringify([name, argTypes, callSite]);
    const cached = this.resolveCacheMap.get(key);
    if (cached !== undefined) return cached;
    const result = resolveFunction(
      name,
      argTypes,
      callSite,
      this.functionIndex
    );
    this.resolveCacheMap.set(key, result);
    return result;
  }

  // ── Compile specialized (resolve + compile) ─────────────────────────

  private compileSpecialized(
    name: string,
    argTypes: ItemType[],
    callSite: CallSite
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ): ((...args: any[]) => any) | null {
    const target = this.resolve(name, argTypes, callSite);
    if (!target || target.kind === "builtin") return null;

    const cKey = JSON.stringify(target);
    const cached = this.compileCacheMap.get(cKey);
    if (cached !== undefined) return cached;

    const fn = this.compileTarget(target);
    this.compileCacheMap.set(cKey, fn);
    return fn;
  }

  // ── Resolve class method ────────────────────────────────────────────

  private resolveClassMethod(
    className: string,
    methodName: string
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ): ((...args: any[]) => any) | null {
    const jitCtx = this.ctx.createJitContext();

    const definingClass = jitCtx.findDefiningClass(className, methodName);
    const classCtx = jitCtx.getOrCreateClassFileContext(definingClass);
    const stub = classCtx?.getLocalFunctionStub(methodName) ?? null;
    if (!classCtx || !stub) return null;

    const selfType: ItemType = { kind: "ClassInstance", className };
    const isStatic = jitCtx.classHasStaticMethod(definingClass, methodName);
    const compileArgTypes: ItemType[] = stub.params.map(
      (_, i) =>
        i === 0 && !isStatic ? selfType : ({ kind: "Unknown" } as ItemType) // todo: is this right?
    );

    const target: ResolvedTarget = {
      kind: "classMethod",
      className,
      methodName,
      compileArgTypes,
      stripInstance: false,
    };

    const cKey = JSON.stringify(target);
    const cached = this.compileCacheMap.get(cKey);
    if (cached !== undefined) return cached;

    const guardKey = `${className}.${methodName}`;
    this.jitInProgress.add(guardKey);
    try {
      const fn = this.compileTarget(target);
      this.compileCacheMap.set(cKey, fn);
      return fn;
    } finally {
      this.jitInProgress.delete(guardKey);
    }
  }

  // ── Core compilation dispatch ───────────────────────────────────────

  private compileTarget(
    target: ResolvedTarget
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ): ((...args: any[]) => any) | null {
    switch (target.kind) {
      case "classMethod":
        return this.compileClassMethod(target);
      case "localFunction":
        return this.compileLocalFunction(target);
      case "privateFunction":
        return this.compilePrivateFunction(target);
      case "workspaceFunction":
        return this.compileWorkspaceFunction(target);
      case "workspaceClassConstructor":
        return this.compileWorkspaceClassConstructor(target);
      case "builtin":
        return null;
    }
  }

  // ── Per-kind compilation methods ────────────────────────────────────

  private static fmtArgs(argTypes: ItemType[]): string {
    return argTypes.map(t => typeToString(t)).join(", ");
  }

  private compileClassMethod(
    target: Extract<ResolvedTarget, { kind: "classMethod" }>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ): ((...args: any[]) => any) | null {
    const jitCtx = this.ctx.createJitContext();
    const desc = `class method: ${target.className}.${target.methodName}(${JitCompiler.fmtArgs(target.compileArgTypes)})`;
    const result = this.tryCompile(
      cg =>
        cg.ensureClassMethodGenerated(
          target.className,
          target.methodName,
          target.compileArgTypes
        ),
      jitCtx,
      desc
    );
    if (!result) return null;
    const fn = this.evaluateAndReturn(result.jsCode, result.jsId);
    if (fn && target.stripInstance) {
      return (nargout: number, ...runtimeArgs: unknown[]) =>
        fn(nargout, ...runtimeArgs.slice(1));
    }
    return fn;
  }

  private compileLocalFunction(
    target: Extract<ResolvedTarget, { kind: "localFunction" }>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ): ((...args: any[]) => any) | null {
    const { source } = target;
    const argStr = JitCompiler.fmtArgs(target.argTypes);

    if (source.from === "main") {
      const desc = `local function: ${target.name}(${argStr})`;
      const result = this.tryCompile(
        cg =>
          cg.ensureSpecializedFunctionGenerated(target.name, target.argTypes),
        undefined,
        desc
      );
      if (!result) return null;
      return this.evaluateAndReturn(result.jsCode, result.jsId);
    }

    if (source.from === "classFile") {
      const jitCtx = this.ctx.createJitContext();
      const classCtx = jitCtx.getOrCreateClassFileContext(source.className);
      if (!classCtx) return null;
      const desc = `local function in class ${source.className}: ${target.name}(${argStr})`;
      const result = this.tryCompile(
        cg => {
          cg.loweringCtx = classCtx;
          const generate = () =>
            cg.ensureSpecializedFunctionGenerated(target.name, target.argTypes);
          if (source.methodScope) {
            cg.currentMethodName = source.methodScope;
            return classCtx.withMethodScope(source.methodScope, generate);
          }
          return generate();
        },
        jitCtx,
        desc
      );
      if (!result) return null;
      return this.evaluateAndReturn(result.jsCode, result.jsId);
    }

    if (source.from === "workspaceFile") {
      const jitCtx = this.ctx.createJitContext();
      const wsCtx = jitCtx.getOrCreateWorkspaceFileContext(source.wsName);
      if (!wsCtx) return null;
      const desc = `local function in workspace file ${source.wsName}: ${target.name}(${argStr})`;
      const result = this.tryCompile(
        cg => {
          cg.loweringCtx = wsCtx;
          return cg.ensureSpecializedFunctionGenerated(
            target.name,
            target.argTypes
          );
        },
        jitCtx,
        desc
      );
      if (!result) return null;
      return this.evaluateAndReturn(result.jsCode, result.jsId);
    }

    // source.from === "privateFile"
    return this.compilePrivateFunctionFromFile(
      source.callerFile,
      source.callerFile.replace(/\.m$/, "").split("/").pop()!,
      target.name,
      target.argTypes
    );
  }

  private compilePrivateFunction(
    target: Extract<ResolvedTarget, { kind: "privateFunction" }>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ): ((...args: any[]) => any) | null {
    return this.compilePrivateFunctionFromFile(
      target.callerFile,
      target.name,
      target.name,
      target.argTypes
    );
  }

  private compilePrivateFunctionFromFile(
    callerFile: string,
    fileLookupName: string,
    name: string,
    argTypes: ItemType[]
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ): ((...args: any[]) => any) | null {
    const jitCtx = this.ctx.createJitContext();
    const privEntry = this.ctx.getPrivateFileEntry(callerFile, fileLookupName);
    if (!privEntry) return null;
    const privAst = this.fileASTCache.get(privEntry.fileName);
    if (!privAst) {
      throw new Error(
        `FileASTCache miss: no cached AST for "${privEntry.fileName}"`
      );
    }
    const privCtx = new LoweringContext(privEntry.source, privEntry.fileName);
    for (const stmt of privAst.body) {
      if (stmt.type === "Function") {
        privCtx.registerLocalFunctionAST(stmt);
      }
    }
    privCtx["registry"] = jitCtx["registry"];
    const desc = `private function: ${name}(${JitCompiler.fmtArgs(argTypes)})`;
    const result = this.tryCompile(
      cg => {
        cg.loweringCtx = privCtx;
        return cg.ensureSpecializedFunctionGenerated(name, argTypes);
      },
      jitCtx,
      desc
    );
    if (!result) return null;
    return this.evaluateAndReturn(result.jsCode, result.jsId);
  }

  private compileWorkspaceFunction(
    target: Extract<ResolvedTarget, { kind: "workspaceFunction" }>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ): ((...args: any[]) => any) | null {
    const jitCtx = this.ctx.createJitContext();
    const desc = `workspace function: ${target.name}(${JitCompiler.fmtArgs(target.argTypes)})`;
    const result = this.tryCompile(
      cg => cg.ensureWorkspaceFunctionGenerated(target.name, target.argTypes),
      jitCtx,
      desc
    );
    if (!result) return null;
    return this.evaluateAndReturn(result.jsCode, result.jsId);
  }

  private compileWorkspaceClassConstructor(
    target: Extract<ResolvedTarget, { kind: "workspaceClassConstructor" }>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ): ((...args: any[]) => any) | null {
    const jitCtx = this.ctx.createJitContext();
    const desc = `workspace class constructor: ${target.className}(${JitCompiler.fmtArgs(target.argTypes)})`;
    const result = this.tryCompile(
      cg =>
        cg.ensureWorkspaceClassConstructorGenerated(
          target.className,
          target.argTypes
        ),
      jitCtx,
      desc
    );
    if (!result) return null;
    return this.evaluateAndReturn(result.jsCode, result.jsId);
  }

  // ── Low-level helpers ───────────────────────────────────────────────

  private tryCompile(
    compile: (cg: Codegen) => string | null,
    lowerCtx: LoweringContext = this.ctx,
    description?: string
  ): { jsCode: string; jsId: string } | null {
    const miniCodegen = new Codegen(lowerCtx, this.fileSources);
    if (this.noLineTracking) {
      miniCodegen.noLineTracking = true;
    }
    let jsId: string | null = null;
    try {
      jsId = compile(miniCodegen);
    } catch {
      return null;
    }
    if (!jsId) return null;
    for (const [id, fnCode] of miniCodegen.perFunctionCode) {
      this.rt.jitFunctionCode.set(id, fnCode);
    }
    const jsCode = miniCodegen.getCode();
    if (this.onJitCompile && description) {
      this.onJitCompile(description, jsCode);
    }
    return { jsCode, jsId };
  }

  private evaluateAndReturn(
    jsCode: string,
    jsId: string
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ): ((...args: any[]) => any) | null {
    const wrappedCode = `${jsCode}\nreturn ${jsId};`;
    const setupFn = new Function("$rt", wrappedCode);
    return setupFn(this.rt);
  }
}
