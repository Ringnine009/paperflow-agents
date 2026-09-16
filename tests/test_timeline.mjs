/* Node unit tests for the execution-timeline + claim-linking helpers. */

import { createRequire } from "module";
import assert from "node:assert";

const require = createRequire(import.meta.url);
const { buildTimeline, linkClaims, normText } = require("../paperflow/web/static/timeline.js");

const T0 = Date.parse("2026-08-22T01:00:00Z");
const MIN = 60 * 1000;

function agent(name, status, startMs, endMs) {
  return {
    status,
    started_at: startMs == null ? undefined : new Date(startMs).toISOString(),
    finished_at: endMs == null ? undefined : new Date(endMs).toISOString(),
  };
}

/* --- buildTimeline: timed --- */
const agents = {
  researcher: agent("researcher", "done", T0, T0 + 2 * MIN),
  reader: agent("reader", "done", T0 + 2 * MIN, T0 + 5 * MIN),
  critic: agent("critic", "done", T0 + 5 * MIN, T0 + 7 * MIN),
  synthesizer: agent("synthesizer", "done", T0 + 7 * MIN, T0 + 9 * MIN),
};
const tl = buildTimeline(agents);
assert(tl.timed === true, "timed timeline");
assert.deepStrictEqual(tl.rows.map(r => r.name), ["researcher", "reader", "critic", "synthesizer"], "agent order");
assert(tl.rows[0].pctLeft === 0, "first agent starts at 0");
assert(tl.rows[0].pctWidth > 0 && tl.rows[0].pctWidth < 100, "first bar sized");
assert(tl.rows[1].pctLeft > tl.rows[0].pctLeft, "bars are sequential");

/* running agent without finished_at extends to `now` */
const running = buildTimeline(
  { researcher: agent("researcher", "done", T0, T0 + 2 * MIN), reader: agent("reader", "running", T0 + 2 * MIN, null) },
  T0 + 4 * MIN
);
assert(running.timed === true, "running still timed");
const readerRow = running.rows[1];
assert(readerRow.end === T0 + 4 * MIN, "running agent extends to now");

/* --- buildTimeline: no timestamps -> flow order fallback --- */
const flow = buildTimeline({});
assert(flow.timed === false, "fallback when no timestamps");
assert(flow.rows.length === 4, "fallback keeps 4 rows");
assert(flow.rows[0].pctLeft === 0 && flow.rows[3].pctLeft === 75, "equal segments");
assert(flow.rows[0].pctWidth === 25, "equal width");

/* --- linkClaims --- */
const blocks = [
  "The unassisted baseline achieves a 44.2% win rate.",
  "Combining MaKTO-Proxy and DBN increases win rate to 68.8%.",
  "Vote accuracy rises monotonically.",
];
const claims = [
  { claim: "The unassisted baseline achieves a 44.2% win rate" },   // matches block 0
  { claim: "This claim never appears in the report" },              // no match
  { claim: "short" },                                               // below min length -> skipped
  { claim: "Vote accuracy rises monotonically" },                   // matches block 2
];
const matches = linkClaims(blocks, claims);
assert.deepStrictEqual(matches, [
  { claimIndex: 0, blockIndex: 0 },
  { claimIndex: 3, blockIndex: 2 },
], "claim -> block mapping");

assert(normText("  a  b\nc ") === "a b c", "normText collapses whitespace");
assert(normText("CLAIM Text") === "claim text", "normText lowercases");

/* --- linkClaims: paraphrased bullets still link ---
   Real case from examples/attention-is-all-you-need: the Reader's claim and
   the Synthesizer's bullet word the same fact differently, which exact
   containment alone cannot link (the public examples were at 0/7). Token
   overlap fixes it, and the thresholds stay strict enough that unrelated
   wording still does not link. */
const readerClaim =
  "The Transformer is the first transduction model relying entirely on self-attention to " +
  "compute representations of its input and output without using sequence-aligned RNNs or convolution";
const reportBullet =
  "- **The Transformer is the first transduction model relying entirely on self-attention " +
  "without sequence-aligned RNNs or convolution.** **[supported]** - the conclusion states it.";
const paraphrased = linkClaims([reportBullet], [{ claim: readerClaim }]);
assert.deepStrictEqual(paraphrased, [{ claimIndex: 0, blockIndex: 0 }], "paraphrased bullet links");

const unrelated = linkClaims(
  ["- **The paper introduces a new optimizer.**"],
  [{ claim: "Vote accuracy rises monotonically across rounds" }]
);
assert.deepStrictEqual(unrelated, [], "unrelated wording does not link");

const competing = linkClaims(
  ["Vote accuracy rises monotonically across rounds."],
  [
    { claim: "Vote accuracy rises monotonically" },
    { claim: "unassisted baseline win rate 44.2% and accuracy rises" },
  ]
);
assert.deepStrictEqual(competing, [{ claimIndex: 0, blockIndex: 0 }], "exact match beats overlap");

/* one block is never claimed twice, and results are ordered by claim index */
const duplicated = linkClaims(
  ["- Vote accuracy rises monotonically across rounds."],
  [{ claim: "Vote accuracy rises monotonically" }, { claim: "accuracy rises monotonically across rounds" }]
);
assert.deepStrictEqual(duplicated, [{ claimIndex: 0, blockIndex: 0 }], "one block, one claim");

console.log("timeline/claim-linking tests OK");
