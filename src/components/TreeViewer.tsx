import { useState, useCallback, createContext, useContext } from "react";
import { Box, Typography } from "@mui/material";
import { offsetToLine } from "../numbl-core/runtime/error.js";

type SpanContext = {
  fileSources: Map<string, string> | null;
};

const SpanCtx = createContext<SpanContext>({
  fileSources: null,
});

type TreeNodeProps = {
  label: string;
  value: unknown;
  depth: number;
  defaultExpanded?: boolean;
};

function formatPrimitive(value: unknown): string {
  if (value === null) return "null";
  if (value === undefined) return "undefined";
  if (typeof value === "string") return `"${value}"`;
  return String(value);
}

function primitiveColor(value: unknown): string {
  if (value === null || value === undefined) return "#569cd6";
  if (typeof value === "string") return "#ce9178";
  if (typeof value === "number") return "#b5cea8";
  if (typeof value === "boolean") return "#569cd6";
  return "#d4d4d4";
}

function resolveSpan(
  span: { file: string; start: number; end: number },
  fileSources: Map<string, string> | null
): string | null {
  const source = fileSources?.get(span.file);
  if (!source) return `${span.file}`;
  const startLine = offsetToLine(source, span.start);
  const endLine = offsetToLine(source, span.end);
  if (startLine === endLine) {
    return `${span.file}:${startLine}`;
  }
  return `${span.file}:${startLine}-${endLine}`;
}

function isSpanObject(label: string, obj: Record<string, unknown>): boolean {
  return (
    label === "span" &&
    typeof obj.file === "string" &&
    typeof obj.start === "number" &&
    typeof obj.end === "number"
  );
}

function SpanNode({
  obj,
  depth,
}: {
  obj: Record<string, unknown>;
  depth: number;
}) {
  const { fileSources } = useContext(SpanCtx);
  const span = obj as unknown as { file: string; start: number; end: number };
  const location = resolveSpan(span, fileSources);

  return (
    <Box
      sx={{ pl: depth * 2, display: "flex", alignItems: "baseline", py: "1px" }}
    >
      <Typography
        component="span"
        sx={{
          fontFamily: "monospace",
          fontSize: "12px",
          color: "#9cdcfe",
          mr: 0.5,
        }}
      >
        span:
      </Typography>
      {location && (
        <Typography
          component="span"
          sx={{
            fontFamily: "monospace",
            fontSize: "12px",
            color: "#4ec9b0",
            mr: 1,
          }}
        >
          {location}
        </Typography>
      )}
      <Typography
        component="span"
        sx={{ fontFamily: "monospace", fontSize: "12px", color: "#808080" }}
      >
        ({span.start}..{span.end})
      </Typography>
    </Box>
  );
}

function TreeNode({ label, value, depth, defaultExpanded }: TreeNodeProps) {
  const [expanded, setExpanded] = useState(defaultExpanded ?? false);
  const toggle = useCallback(() => setExpanded(e => !e), []);

  // Primitives
  if (value === null || value === undefined || typeof value !== "object") {
    return (
      <Box
        sx={{
          pl: depth * 2,
          display: "flex",
          alignItems: "baseline",
          py: "1px",
        }}
      >
        <Typography
          component="span"
          sx={{
            fontFamily: "monospace",
            fontSize: "12px",
            color: "#9cdcfe",
            mr: 0.5,
          }}
        >
          {label}:
        </Typography>
        <Typography
          component="span"
          sx={{
            fontFamily: "monospace",
            fontSize: "12px",
            color: primitiveColor(value),
          }}
        >
          {formatPrimitive(value)}
        </Typography>
      </Box>
    );
  }

  // Arrays
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return (
        <Box sx={{ pl: depth * 2, py: "1px" }}>
          <Typography
            component="span"
            sx={{
              fontFamily: "monospace",
              fontSize: "12px",
              color: "#9cdcfe",
              mr: 0.5,
            }}
          >
            {label}:
          </Typography>
          <Typography
            component="span"
            sx={{ fontFamily: "monospace", fontSize: "12px", color: "#808080" }}
          >
            []
          </Typography>
        </Box>
      );
    }

    return (
      <Box>
        <Box
          sx={{
            pl: depth * 2,
            cursor: "pointer",
            "&:hover": { bgcolor: "rgba(255,255,255,0.04)" },
            py: "1px",
            display: "flex",
            alignItems: "baseline",
          }}
          onClick={toggle}
        >
          <Typography
            component="span"
            sx={{
              fontFamily: "monospace",
              fontSize: "12px",
              color: "#808080",
              mr: 0.5,
              userSelect: "none",
              width: "1ch",
            }}
          >
            {expanded ? "▾" : "▸"}
          </Typography>
          <Typography
            component="span"
            sx={{
              fontFamily: "monospace",
              fontSize: "12px",
              color: "#9cdcfe",
              mr: 0.5,
            }}
          >
            {label}:
          </Typography>
          <Typography
            component="span"
            sx={{ fontFamily: "monospace", fontSize: "12px", color: "#808080" }}
          >
            [{value.length} items]
          </Typography>
        </Box>
        {expanded &&
          value.map((item, i) => (
            <TreeNode
              key={i}
              label={String(i)}
              value={item}
              depth={depth + 1}
            />
          ))}
      </Box>
    );
  }

  // Objects
  const obj = value as Record<string, unknown>;
  const entries = Object.entries(obj);

  // Compact span rendering with file:line resolution
  if (isSpanObject(label, obj)) {
    return <SpanNode obj={obj} depth={depth} />;
  }

  // Object with type field - use type as label hint
  const typeHint = typeof obj.type === "string" ? obj.type : null;

  return (
    <Box>
      <Box
        sx={{
          pl: depth * 2,
          cursor: "pointer",
          "&:hover": { bgcolor: "rgba(255,255,255,0.04)" },
          py: "1px",
          display: "flex",
          alignItems: "baseline",
        }}
        onClick={toggle}
      >
        <Typography
          component="span"
          sx={{
            fontFamily: "monospace",
            fontSize: "12px",
            color: "#808080",
            mr: 0.5,
            userSelect: "none",
            width: "1ch",
          }}
        >
          {expanded ? "▾" : "▸"}
        </Typography>
        <Typography
          component="span"
          sx={{
            fontFamily: "monospace",
            fontSize: "12px",
            color: "#9cdcfe",
            mr: 0.5,
          }}
        >
          {label}:
        </Typography>
        {typeHint && (
          <Typography
            component="span"
            sx={{ fontFamily: "monospace", fontSize: "12px", color: "#dcdcaa" }}
          >
            {typeHint}
          </Typography>
        )}
        {!typeHint && (
          <Typography
            component="span"
            sx={{ fontFamily: "monospace", fontSize: "12px", color: "#808080" }}
          >
            {"{…}"}
          </Typography>
        )}
      </Box>
      {expanded &&
        entries.map(([key, val]) => (
          <TreeNode key={key} label={key} value={val} depth={depth + 1} />
        ))}
    </Box>
  );
}

export function TreeViewer({
  data,
  label,
  fileSources,
}: {
  data: unknown;
  label?: string;
  fileSources?: Map<string, string> | null;
}) {
  if (data === null || data === undefined) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography
          sx={{ fontFamily: "monospace", fontSize: "12px", color: "#808080" }}
        >
          No data available. Run a script first.
        </Typography>
      </Box>
    );
  }

  return (
    <SpanCtx.Provider
      value={{
        fileSources: fileSources ?? null,
      }}
    >
      <Box sx={{ p: 1, overflow: "auto", height: "100%" }}>
        <TreeNode
          label={label ?? "root"}
          value={data}
          depth={0}
          defaultExpanded
        />
      </Box>
    </SpanCtx.Provider>
  );
}
