import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const interfaceFiles = [
  "../index.html",
  "../src/App.jsx",
  "../src/api.js",
  "../src/components/HealthBar.jsx",
  "../src/components/ResultsTable.jsx",
];

function read(relativePath) {
  return readFileSync(new URL(relativePath, import.meta.url), "utf8");
}

test("system-owned interface copy is English", () => {
  const hanCharacters = /[\u3400-\u4dbf\u4e00-\u9fff]/u;

  for (const relativePath of interfaceFiles) {
    assert.doesNotMatch(read(relativePath), hanCharacters, relativePath);
  }
});

test("CareNL branding and English document metadata are present", () => {
  const app = read("../src/App.jsx");
  const html = read("../index.html");

  assert.match(app, /CareNL: Care New Liver, Care New Life/);
  assert.match(html, /<html lang="en">/);
  assert.match(html, /<title>CareNL \| Pediatric Liver Transplant Risk Prediction<\/title>/);
});
