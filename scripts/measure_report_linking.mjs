/* Offline measurement: how many archived report bullets link to a claim?

The dashboard injects "[#n] go to verification" markers by matching a report
bullet's text against the Reader's claim wording. The Synthesizer paraphrases,
so exact containment alone missed most bullets - including 0/7 on the
`attention-is-all-you-need` sample. This script replays every archived run
through the old rule and the shipped rule so the difference is measured.

Usage:  node scripts/measure_report_linking.mjs

Offline: it only reads the committed reports and artifacts.
*/

import { createRequire } from "module";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const { linkClaims } = require(path.join(repo, "paperflow", "web", "static", "timeline.js"));

const normText = (s) => String(s == null ? "" : s).replace(/\s+/g, " ").trim().toLowerCase();

/* the pre-fix rule: exact normalized containment only */
function exactLink(blocks, claims, minLen = 6) {
  const matches = [];
  claims.forEach((claim, index) => {
    const text = claim && claim.claim ? normText(claim.claim) : "";
    if (text.length < minLen) return;
    for (let block = 0; block < blocks.length; block++) {
      if (normText(blocks[block]).indexOf(text) !== -1) {
        matches.push({ claimIndex: index, blockIndex: block });
        break;
      }
    }
  });
  return matches;
}

function bulletsOf(reportText) {
  const lines = reportText.split(/\r?\n/);
  const start = lines.findIndex((line) => line.trim() === "## Key Claims & Evidence");
  if (start === -1) return [];
  const bullets = [];
  for (let i = start + 1; i < lines.length; i++) {
    if (/^##\s/.test(lines[i])) break;
    if (/^\s{0,3}[-*]\s+\S/.test(lines[i])) bullets.push(lines[i]);
  }
  return bullets;
}

let total = 0;
let exact = 0;
let shipped = 0;
const rows = [];

for (const root of ["outputs", "examples"]) {
  const dir = path.join(repo, root);
  if (!fs.existsSync(dir)) continue;
  for (const run of fs.readdirSync(dir).sort()) {
    const report = path.join(dir, run, "report.md");
    const reader = path.join(dir, run, "artifacts", "reader_output.json");
    if (!fs.existsSync(report) || !fs.existsSync(reader)) continue;
    const blocks = bulletsOf(fs.readFileSync(report, "utf8"));
    const claims = JSON.parse(fs.readFileSync(reader, "utf8")).claims || [];
    if (!blocks.length || !claims.length) continue;
    const before = exactLink(blocks, claims).length;
    const after = linkClaims(blocks, claims).length;
    total += blocks.length;
    exact += before;
    shipped += after;
    rows.push(`${run.padEnd(30)} bullets=${blocks.length} exact=${before} shipped=${after}`);
  }
}

const pct = (n) => `${((100 * n) / total).toFixed(0)}%`;
console.log(rows.join("\n"));
console.log(`\nbullets=${total}  exact-only=${exact} (${pct(exact)})  shipped=${shipped} (${pct(shipped)})`);
