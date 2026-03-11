import { useState, useEffect, useRef } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Typography,
} from "@mui/material";
import { useNavigate } from "react-router-dom";
import { validateProjectName } from "../utils/validation";
import { createProject, listProjects } from "../db/operations";

interface CreateProjectDialogProps {
  open: boolean;
  onClose: () => void;
  onCreated?: () => void;
}

export function CreateProjectDialog({
  open,
  onClose,
  onCreated,
}: CreateProjectDialogProps) {
  const [projectName, setProjectName] = useState("");
  const [error, setError] = useState<string>("");
  const [creating, setCreating] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    if (!open) return;
    listProjects().then(projects => {
      const existingNames = new Set(projects.map(p => p.name));
      let name = "untitled";
      let i = 2;
      while (existingNames.has(name)) {
        name = `untitled-${i++}`;
      }
      setProjectName(name);
    });
  }, [open]);

  const inputElRef = useRef<HTMLInputElement | null>(null);

  // Select all text once the initial project name is populated
  useEffect(() => {
    if (open && projectName && inputElRef.current) {
      inputElRef.current.select();
    }
  }, [open, projectName === ""]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setProjectName(value);
    // Clear error when user types
    if (error) setError("");
  };

  const handleCreate = async () => {
    // Validate project name
    const projects = await listProjects();
    const existingNames = projects.map(p => p.name);
    const validation = validateProjectName(projectName, existingNames);

    if (!validation.valid) {
      setError(validation.error || "Invalid project name");
      return;
    }

    setCreating(true);

    try {
      await createProject(projectName);
      // Navigate to new project
      navigate(`/project/${projectName}`);
      // Close dialog
      handleClose();
      // Notify parent
      onCreated?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create project");
    } finally {
      setCreating(false);
    }
  };

  const handleClose = () => {
    if (creating) return; // Don't close while creating
    setProjectName("");
    setError("");
    onClose();
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !creating) {
      handleCreate();
    }
  };

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>Create New Project</DialogTitle>
      <DialogContent>
        <TextField
          autoFocus
          inputRef={inputElRef}
          margin="dense"
          label="Project Name"
          type="text"
          fullWidth
          variant="outlined"
          value={projectName}
          onChange={handleChange}
          onKeyPress={handleKeyPress}
          error={!!error}
          helperText={
            error ||
            "Use only letters, numbers, dashes, and underscores (no spaces)"
          }
          disabled={creating}
        />
        {error && (
          <Typography variant="caption" color="error" sx={{ mt: 1 }}>
            {error}
          </Typography>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose} disabled={creating}>
          Cancel
        </Button>
        <Button
          onClick={handleCreate}
          variant="contained"
          disabled={!projectName.trim() || creating}
        >
          {creating ? "Creating..." : "Create"}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
