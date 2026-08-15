import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  displayRationale,
  formatProbability,
  normalizePatterns,
  resultStatusClass,
} from "../src/patterns.js";

test("normalizes the backend patterns array", () => {
  assert.deepEqual(
    normalizePatterns({
      patterns: [{ pattern: "ALB@异常", analysis: "白蛋白偏低。" }],
    }),
    [{ pattern: "ALB@异常", analysis: "白蛋白偏低。" }]
  );
});

test("accepts compatibility field names and string items", () => {
  assert.deepEqual(
    normalizePatterns({
      pattern_pairs: [
        { Pattern: "ALT@改善", Analysis: "转氨酶下降。" },
        "INR@正常",
      ],
    }),
    [
      { pattern: "ALT@改善", analysis: "转氨酶下降。" },
      { pattern: "INR@正常", analysis: "" },
    ]
  );
});

test("parses a JSON-encoded patterns array", () => {
  assert.deepEqual(
    normalizePatterns({
      patterns: '[{"pattern":"TB@异常","analysis":"胆红素升高。"}]',
    }),
    [{ pattern: "TB@异常", analysis: "胆红素升高。" }]
  );
});

test("falls back to tagged pattern output", () => {
  assert.deepEqual(
    normalizePatterns({
      pattern_output:
        "<Pattern>WBC@波动</Pattern><Analysis>白细胞计数存在波动。</Analysis>",
    }),
    [{ pattern: "WBC@波动", analysis: "白细胞计数存在波动。" }]
  );
});

test("displays rationale only and never falls back to raw Answer JSON", () => {
  assert.equal(
    displayRationale({
      answer_text: '{"1m":0.0129,"1y":null,"5y":null}',
      rationale: "胆红素和凝血指标异常提示总体风险。",
    }),
    "胆红素和凝血指标异常提示总体风险。"
  );
  assert.equal(displayRationale({ answer_text: '{"1m":0.1}' }), "");
});

test("renders partial as a neutral status", () => {
  assert.equal(resultStatusClass("ok"), "tag tag-ok");
  assert.equal(resultStatusClass("partial"), "tag tag-muted");
  assert.equal(resultStatusClass("disabled"), "tag tag-muted");
  assert.equal(resultStatusClass("error"), "tag tag-bad");
});

test("formats explicit model null separately from unavailable values", () => {
  assert.equal(formatProbability(0.1234), "12.34%");
  assert.equal(formatProbability(null, { showNull: true }), "null");
  assert.equal(formatProbability(null), "—");
  assert.equal(formatProbability(undefined, { showNull: true }), "—");
  assert.equal(formatProbability(Number.NaN, { showNull: true }), "—");
});

test("results UI contains no probability completion source labels", () => {
  const source = readFileSync(
    new URL("../src/components/ResultsTable.jsx", import.meta.url),
    "utf8"
  );
  assert.equal(source.includes("ML 回填"), false);
  assert.equal(source.includes("系统补全"), false);
  assert.equal(source.includes("probability_sources"), false);
  assert.equal(source.includes("source-badge"), false);
  assert.equal(source.includes("parse-status-warning"), false);
});
