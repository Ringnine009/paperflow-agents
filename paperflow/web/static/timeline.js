/* PaperFlow execution-timeline + report-claim linking helpers.

Extracted from index.html so the pure logic is unit-testable with Node.
Works as a browser global (window.PaperFlowTL) and as a CommonJS module.

buildTimeline: turn board.agents (started_at/finished_at) into positioned
bar rows for a horizontal timeline; degrades to a flow-order layout when no
timestamps exist.

linkClaims: deterministically find which report blocks contain each claim's
text, so the UI can inject clickable "go to verification" markers. Exact
containment is tried first (score 1); a bullet the Synthesizer paraphrased is
linked by token overlap instead, which is what the machine ledger on the
Python side uses too (97% of archived bullets align, against 0/7 with exact
matching only).
*/

(function (root, factory) {
  if (typeof module !== "undefined" && module.exports) {
    module.exports = factory();
  } else {
    root.PaperFlowTL = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var ORDER = ["researcher", "reader", "critic", "synthesizer"];

  //: a claim needs at least this many distinct tokens before overlap is trusted
  var FUZZY_MIN_TOKENS = 4;
  //: share of the claim's tokens a block must contain to be linked by overlap
  var FUZZY_THRESHOLD = 0.6;

  function normText(s) {
    return String(s == null ? "" : s).replace(/\s+/g, " ").trim().toLowerCase();
  }

  function tokens(s) {
    return normText(s).split(/[^0-9a-z]+/).filter(function (t) {
      return t.length > 0;
    });
  }

  function uniqueTokens(s) {
    var seen = {};
    return tokens(s).filter(function (t) {
      if (seen[t]) return false;
      seen[t] = true;
      return true;
    });
  }

  /** 1 for an exact (normalized) containment, else the claim-token share. */
  function linkScore(claimText, blockText) {
    var claim = normText(claimText);
    if (claim.length > 0 && normText(blockText).indexOf(claim) !== -1) return 1;
    var claimTokens = uniqueTokens(claimText);
    if (claimTokens.length < FUZZY_MIN_TOKENS) return 0;
    var blockTokens = {};
    tokens(blockText).forEach(function (t) {
      blockTokens[t] = true;
    });
    var hit = 0;
    claimTokens.forEach(function (t) {
      if (blockTokens[t]) hit += 1;
    });
    return hit / claimTokens.length;
  }

  function buildTimeline(agents, now) {
    agents = agents || {};
    var rows = [];
    var hasTimes = false;
    ORDER.forEach(function (name) {
      var a = agents[name] || { status: "pending" };
      var start = a.started_at ? Date.parse(a.started_at) : NaN;
      var end = a.finished_at ? Date.parse(a.finished_at) : NaN;
      if (isNaN(end) && !isNaN(start) && a.status === "running") end = now || Date.now();
      if (!isNaN(start) && !isNaN(end)) hasTimes = true;
      rows.push({ name: name, status: a.status || "pending", start: start, end: end });
    });

    if (!hasTimes) {
      // graceful degradation: flow order, equal segments, no real timing
      var w = 100 / ORDER.length;
      return {
        timed: false,
        rows: rows.map(function (r, i) {
          return { name: r.name, status: r.status, pctLeft: i * w, pctWidth: w };
        }),
      };
    }

    var min = Infinity, max = -Infinity;
    rows.forEach(function (r) {
      if (!isNaN(r.start)) min = Math.min(min, r.start);
      if (!isNaN(r.end)) max = Math.max(max, r.end);
    });
    if (!isFinite(min) || !isFinite(max) || max - min < 1000) max = min + 1000;

    rows.forEach(function (r) {
      if (isNaN(r.start) || isNaN(r.end)) {
        r.pctLeft = 0;
        r.pctWidth = 0; // waiting / no timing -> no bar
        return;
      }
      r.pctLeft = Math.max(0, ((r.start - min) / (max - min)) * 100);
      r.pctWidth = Math.max(3, ((r.end - r.start) / (max - min)) * 100);
      if (r.pctLeft + r.pctWidth > 100) r.pctWidth = Math.max(3, 100 - r.pctLeft);
    });
    return { timed: true, rows: rows };
  }

  function linkClaims(blockTexts, claims, minLen) {
    minLen = minLen || 6;
    var blocks = blockTexts || [];
    var candidates = [];
    (claims || []).forEach(function (c, i) {
      var claimText = c && c.claim ? c.claim : "";
      if (normText(claimText).length < minLen) return;
      for (var b = 0; b < blocks.length; b++) {
        var score = linkScore(claimText, blocks[b]);
        if (score >= FUZZY_THRESHOLD) candidates.push({ score: score, claimIndex: i, blockIndex: b });
      }
    });
    // highest score wins; a claim and a block are each claimed at most once
    candidates.sort(function (x, y) {
      return y.score - x.score || x.claimIndex - y.claimIndex || x.blockIndex - y.blockIndex;
    });
    var usedClaims = {}, usedBlocks = {}, matches = [];
    candidates.forEach(function (c) {
      if (usedClaims[c.claimIndex] || usedBlocks[c.blockIndex]) return;
      usedClaims[c.claimIndex] = true;
      usedBlocks[c.blockIndex] = true;
      matches.push({ claimIndex: c.claimIndex, blockIndex: c.blockIndex });
    });
    matches.sort(function (x, y) {
      return x.claimIndex - y.claimIndex;
    });
    return matches;
  }

  return { buildTimeline: buildTimeline, linkClaims: linkClaims, normText: normText };
});
