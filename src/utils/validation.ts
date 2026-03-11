export interface ValidationResult {
  valid: boolean;
  error?: string;
}

export function validateProjectName(
  name: string,
  existingNames: string[] = []
): ValidationResult {
  // Empty check
  if (!name.trim()) {
    return { valid: false, error: "Project name cannot be empty" };
  }

  // No spaces
  if (/\s/.test(name)) {
    return { valid: false, error: "Project name cannot contain spaces" };
  }

  // Valid characters (alphanumeric, dash, underscore)
  if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
    return {
      valid: false,
      error:
        "Project name can only contain letters, numbers, dashes, and underscores",
    };
  }

  // Length check
  if (name.length < 1 || name.length > 50) {
    return { valid: false, error: "Project name must be 1-50 characters" };
  }

  // Uniqueness check
  if (existingNames.includes(name)) {
    return { valid: false, error: "A project with this name already exists" };
  }

  // Reserved names
  const reserved = [
    "project",
    "new",
    "create",
    "delete",
    "admin",
    "settings",
    "share",
  ];
  if (reserved.includes(name.toLowerCase())) {
    return { valid: false, error: "This project name is reserved" };
  }

  return { valid: true };
}
