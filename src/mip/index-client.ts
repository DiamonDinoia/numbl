import {
  PackageIndex,
  PackageIndexEntry,
  MipArchitecture,
  MipBackend,
} from "./types.js";

const DEFAULT_INDEX_URL = "https://mip-org.github.io/mip-core/index.json";

export async function fetchPackageIndex(
  backend: MipBackend,
  indexUrl: string = DEFAULT_INDEX_URL
): Promise<PackageIndex> {
  return backend.fetchJson<PackageIndex>(indexUrl);
}

/** Find the best entry for a package on the given architecture (prefers exact match over "any") */
export function findPackageEntry(
  index: PackageIndex,
  packageName: string,
  arch: MipArchitecture
): PackageIndexEntry | undefined {
  const exact = index.packages.find(
    p => p.name === packageName && p.architecture === arch
  );
  if (exact) return exact;
  return index.packages.find(
    p => p.name === packageName && p.architecture === "any"
  );
}

export function listAvailablePackages(
  index: PackageIndex,
  arch: MipArchitecture
): PackageIndexEntry[] {
  return index.packages.filter(
    p => p.architecture === arch || p.architecture === "any"
  );
}
