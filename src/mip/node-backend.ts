import { MipBackend, MipArchitecture } from "./types.js";
import {
  existsSync,
  readFileSync,
  writeFileSync,
  mkdirSync,
  readdirSync,
  statSync,
  rmSync,
} from "fs";
import { dirname, join } from "path";
import { homedir, tmpdir } from "os";
import { execSync } from "child_process";

export class NodeMipBackend implements MipBackend {
  getMipDir(): string {
    return process.env.MIP_DIR || join(homedir(), ".mip");
  }

  getArchitecture(): MipArchitecture {
    const platform = process.platform;
    const arch = process.arch;

    if (platform === "linux" && arch === "x64") return "linux_x86_64";
    if (platform === "darwin" && arch === "arm64") return "macos_arm64";
    if (platform === "darwin" && arch === "x64") return "macos_x86_64";
    if (platform === "win32" && arch === "x64") return "windows_x86_64";

    throw new Error(`Unsupported platform/architecture: ${platform}/${arch}`);
  }

  async dirExists(path: string): Promise<boolean> {
    try {
      return existsSync(path) && statSync(path).isDirectory();
    } catch {
      return false;
    }
  }

  async readTextFile(path: string): Promise<string | null> {
    try {
      return readFileSync(path, "utf-8");
    } catch {
      return null;
    }
  }

  async writeTextFile(path: string, content: string): Promise<void> {
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, content, "utf-8");
  }

  async readJsonFile<T>(path: string): Promise<T | null> {
    const text = await this.readTextFile(path);
    if (text === null) return null;
    return JSON.parse(text) as T;
  }

  async listDirs(path: string): Promise<string[]> {
    try {
      const entries = readdirSync(path);
      return entries.filter(e => {
        try {
          return statSync(join(path, e)).isDirectory();
        } catch {
          return false;
        }
      });
    } catch {
      return [];
    }
  }

  async removeDir(path: string): Promise<void> {
    rmSync(path, { recursive: true, force: true });
  }

  async fetchJson<T>(url: string): Promise<T> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(
        `Failed to fetch ${url}: ${response.status} ${response.statusText}`
      );
    }
    return (await response.json()) as T;
  }

  async downloadAndExtractZip(url: string, targetDir: string): Promise<void> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to download ${url}: ${response.status}`);
    }
    const buffer = Buffer.from(await response.arrayBuffer());

    const tmpFile = join(tmpdir(), `mip-${Date.now()}.zip`);
    try {
      writeFileSync(tmpFile, buffer);
      mkdirSync(targetDir, { recursive: true });
      execSync(`unzip -o -q "${tmpFile}" -d "${targetDir}"`);
    } finally {
      try {
        rmSync(tmpFile);
      } catch {
        /* ignore cleanup errors */
      }
    }
  }
}
