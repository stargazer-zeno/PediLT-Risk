export function normalizePatterns(llm) {
  const source = llm?.patterns ?? llm?.pattern_pairs;

  if (Array.isArray(source)) {
    return source
      .map((item) => {
        if (typeof item === "string") {
          return { pattern: item.trim(), analysis: "" };
        }
        return {
          pattern: String(item?.pattern ?? item?.Pattern ?? "").trim(),
          analysis: String(item?.analysis ?? item?.Analysis ?? "").trim(),
        };
      })
      .filter((item) => item.pattern || item.analysis);
  }

  if (typeof source === "string" && source.trim()) {
    try {
      const parsed = JSON.parse(source);
      if (Array.isArray(parsed)) {
        return normalizePatterns({ patterns: parsed });
      }
    } catch {
      return [{ pattern: source.trim(), analysis: "" }];
    }
  }

  const raw = llm?.pattern_output;
  if (typeof raw !== "string" || !raw.trim()) return [];

  const pairs = [];
  const pairPattern =
    /<Pattern>([\s\S]*?)<\/Pattern>\s*<Analysis>([\s\S]*?)<\/Analysis>/gi;
  for (const match of raw.matchAll(pairPattern)) {
    pairs.push({
      pattern: match[1].trim(),
      analysis: match[2].trim(),
    });
  }
  return pairs;
}

export function displayRationale(llm) {
  return typeof llm?.rationale === "string" ? llm.rationale.trim() : "";
}

export function formatProbability(value, { showNull = false } = {}) {
  if (value === null) return showNull ? "null" : "—";
  if (value === undefined || typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }
  return `${(value * 100).toFixed(2)}%`;
}

export function resultStatusClass(status) {
  if (status === "ok") return "tag tag-ok";
  if (status === "disabled" || status === "partial") return "tag tag-muted";
  return "tag tag-bad";
}
