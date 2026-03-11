import { MipManager } from "./mip/mip-manager.js";
import { NodeMipBackend } from "./mip/node-backend.js";

function printMipHelp(): void {
  console.log(`Usage: numbl mip <subcommand> [options]

Subcommands:
  install <package>    Install a package and its dependencies
  uninstall <package>  Uninstall a package and prune orphaned dependencies
  list                 List installed packages
  avail                List packages available for this platform
  info <package>       Show details about a package`);
}

export async function cmdMip(args: string[]): Promise<void> {
  if (args.length === 0 || args[0] === "--help" || args[0] === "-h") {
    printMipHelp();
    return;
  }

  const backend = new NodeMipBackend();
  const manager = new MipManager(backend);
  const subcommand = args[0];
  const rest = args.slice(1);

  switch (subcommand) {
    case "install":
      await mipInstall(manager, rest);
      break;
    case "uninstall":
      await mipUninstall(manager, rest);
      break;
    case "list":
      await mipList(manager);
      break;
    case "avail":
      await mipAvail(manager);
      break;
    case "info":
      await mipInfo(manager, rest);
      break;
    default:
      console.error(`Unknown mip subcommand: ${subcommand}`);
      console.error('Run "numbl mip --help" for usage.');
      process.exit(1);
  }
}

async function mipInstall(manager: MipManager, args: string[]): Promise<void> {
  if (args.length !== 1) {
    console.error("Usage: numbl mip install <package-name>");
    process.exit(1);
  }

  try {
    const result = await manager.install(args[0]);
    if (result.installed.length === 0) {
      console.log(
        `${args[0]} is already installed (along with all dependencies).`
      );
    } else {
      console.log(`\nInstalled: ${result.installed.join(", ")}`);
      if (result.alreadyInstalled.length > 0) {
        console.log(`Already present: ${result.alreadyInstalled.join(", ")}`);
      }
    }
  } catch (error) {
    console.error(`Error: ${(error as Error).message}`);
    process.exit(1);
  }
}

async function mipUninstall(
  manager: MipManager,
  args: string[]
): Promise<void> {
  if (args.length !== 1) {
    console.error("Usage: numbl mip uninstall <package-name>");
    process.exit(1);
  }

  try {
    const result = await manager.uninstall(args[0]);
    if (result.notInstalled) {
      console.error(`Package "${args[0]}" is not installed.`);
      process.exit(1);
    }
    console.log(`Removed: ${result.removed.join(", ")}`);
    if (result.pruned.length > 0) {
      console.log(`Pruned orphaned dependencies: ${result.pruned.join(", ")}`);
    }
  } catch (error) {
    console.error(`Error: ${(error as Error).message}`);
    process.exit(1);
  }
}

async function mipList(manager: MipManager): Promise<void> {
  const packages = await manager.list();
  if (packages.length === 0) {
    console.log("No packages installed.");
    return;
  }

  console.log("Installed packages:\n");
  for (const pkg of packages) {
    const marker = pkg.isDirect ? " (direct)" : " (dependency)";
    console.log(`  ${pkg.name} v${pkg.version}${marker}`);
  }
}

async function mipAvail(manager: MipManager): Promise<void> {
  const packages = await manager.avail();
  if (packages.length === 0) {
    console.log("No packages available for this architecture.");
    return;
  }

  console.log(`Available packages:\n`);
  for (const pkg of packages) {
    const desc = pkg.description ? ` - ${pkg.description}` : "";
    console.log(`  ${pkg.name} v${pkg.version}${desc}`);
  }
}

async function mipInfo(manager: MipManager, args: string[]): Promise<void> {
  if (args.length !== 1) {
    console.error("Usage: numbl mip info <package-name>");
    process.exit(1);
  }

  const { installed, available } = await manager.info(args[0]);

  if (!installed && !available) {
    console.error(
      `Package "${args[0]}" not found (not installed and not in index).`
    );
    process.exit(1);
  }

  if (installed) {
    console.log("Installed:");
    console.log(`  Name:         ${installed.name}`);
    console.log(`  Version:      ${installed.version}`);
    console.log(
      `  Type:         ${installed.isDirect ? "directly installed" : "dependency"}`
    );
    console.log(
      `  Dependencies: ${installed.mipJson.dependencies.join(", ") || "none"}`
    );
    console.log(
      `  Symbols:      ${installed.mipJson.exposed_symbols.length} exposed`
    );
  }

  if (available) {
    if (installed) console.log("");
    console.log("Available in index:");
    console.log(`  Name:         ${available.name}`);
    console.log(`  Version:      ${available.version}`);
    console.log(`  Architecture: ${available.architecture}`);
    console.log(`  License:      ${available.license}`);
    console.log(
      `  Dependencies: ${available.dependencies.join(", ") || "none"}`
    );
    console.log(`  Homepage:     ${available.homepage}`);
    console.log(`  Symbols:      ${available.exposed_symbols.length} exposed`);
  }
}
