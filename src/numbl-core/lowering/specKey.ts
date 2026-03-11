/**
 * Specialization key computation.
 *
 * Shared between the lowering and codegen layers for cache keying
 * of specialized function versions by (name, argTypes).
 */

import { type ItemType } from "../lowering/itemTypes.js";

/** Convert a type to a JSON-serializable descriptor for specialization keys. */
function typeToDescriptor(
  ty: ItemType | undefined
): { kind: string } | { kind: "ClassInstance"; className: string } {
  if (!ty) return { kind: "Unknown" };
  if (ty.kind === "ClassInstance") {
    return { kind: "ClassInstance", className: ty.className };
  }
  return { kind: ty.kind };
}

/**
 * Compute a specialization cache key as a deterministic JSON string.
 * Captures full type info including class names for ClassInstance.
 */
export function computeSpecKey(name: string, argTypes: ItemType[]): string {
  const args = argTypes.map(t => typeToDescriptor(t));
  return JSON.stringify({ name, args });
}

/**
 * Compute a short hash string from argument types for use in JS function IDs.
 * Uses FNV-1a to produce an 8-character hex string.
 */
export function hashForJsId(argTypes: ItemType[]): string {
  const args = argTypes.map(t => typeToDescriptor(t));
  const input = JSON.stringify(args);
  let hash = 0x811c9dc5; // FNV offset basis
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193); // FNV prime
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}
