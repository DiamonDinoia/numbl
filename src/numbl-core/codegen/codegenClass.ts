/**
 * Class-related code generation.
 *
 * Handles generating JS code for class methods, constructors,
 * and class registration.
 */

import { type ItemType } from "../lowering/itemTypes.js";
import type { Codegen } from "./codegen.js";
import { computeSpecKey, hashForJsId } from "./codegenHelpers.js";
import { genFunctionDef } from "./codegenFunction.js";
import { type LoweringContext } from "../lowering/loweringContext.js";

/**
 * Ensure a specialized class method (or constructor) has been generated.
 * Similar to ensureWorkspaceFunctionGenerated but uses the class file context.
 */
export function ensureClassMethodGenerated(
  cg: Codegen,
  className: string,
  methodName: string,
  argTypes: ItemType[]
): string | null {
  const definingClass = cg.loweringCtx.findDefiningClass(className, methodName);

  const specKey = computeSpecKey(`${className}.${methodName}`, argTypes);
  const hash = hashForJsId(argTypes);
  const jsId = `$fn_${cg.sanitizeName(definingClass)}$${cg.sanitizeName(methodName)}$${hash}`;
  let classCtx: LoweringContext | null = null;

  return cg.ensureGenerated({
    specKey,
    jsId,
    lower: () => {
      // Get the defining class file's context
      classCtx = cg.loweringCtx.getOrCreateClassFileContext(definingClass);
      if (!classCtx) return null;

      // Add class file source for line tracking
      if (!cg.fileSources.has(classCtx.mainFileName)) {
        cg.fileSources.set(classCtx.mainFileName, classCtx.fileSource);
      }

      // Add external method file source for line tracking (if applicable)
      const classInfo2 = cg.loweringCtx.getClassInfo(definingClass);
      if (classInfo2) {
        const extFile = classInfo2.externalMethodFiles.get(methodName);
        if (extFile && !cg.fileSources.has(extFile.fileName)) {
          cg.fileSources.set(extFile.fileName, extFile.source);
        }
      }

      // Lower the method via the defining class context
      return cg.loweringCtx.getOrLowerClassMethodSpecialized(
        definingClass,
        methodName,
        argTypes
      );
    },
    generate: funcIR => {
      // Constructor has obj prepended as hidden first parameter, so nargin
      // needs an extra offset of 1 to exclude it from the count.
      const classInfo2b = classCtx!.getClassInfo(definingClass);
      const isConstructor = classInfo2b?.constructorName === methodName;

      classCtx!.withMethodScope(methodName, () => {
        cg.withCodegenContext(
          {
            loweringCtx: classCtx!,
            currentMethodName: methodName,
            narginAdjust: isConstructor ? 1 : 0,
          },
          () => {
            genFunctionDef(cg, funcIR, jsId);
          }
        );
      });

      // Clear codegen caches for per-file local functions
      const localHelpers =
        classCtx!.externalMethodLocalFunctions.get(methodName);
      if (localHelpers) {
        for (const helper of localHelpers) {
          cg.clearGeneratedFunction(helper.name);
        }
      }
    },
  });
}

/**
 * Mark a class as registered (no-op for code generation).
 * Methods are generated on demand when called, not eagerly.
 */
export function ensureClassRegistered(cg: Codegen, className: string): void {
  if (cg.isClassRegistered(className)) return;
  cg.markClassRegistered(className);
}
