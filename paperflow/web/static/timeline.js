/* PaperFlow execution-timeline + report-claim linking helpers.

Extracted from index.html so the pure logic is unit-testable with Node.
Works as a browser global (window.PaperFlowTL) and as a CommonJS module.

buildTimeline: turn board.agents (started_at/finished_at) into positioned
bar rows for a horizontal timeline; degrades to a flow-order layout when no
timestamps exist.

linkClaims: deterministically find which report blocks contain each claim's
text, so the UI can inject clickable "go to verification" markers.
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

  function normText(s) {
    return String(s == null ? "" : s).replace(/\s+/g, " ").trim().toLowerCase();
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
    var matches = [];
    (claims || []).forEach(function (c, i) {
      var claimText = c && c.claim ? normText(c.claim) : "";
      if (claimText.length < minLen) return;
      for (var b = 0; b < blockTexts.length; b++) {
        if (normText(blockTexts[b]).indexOf(claimText) !== -1) {
          matches.push({ claimIndex: i, blockIndex: b });
          break;
        }
      }
    });
    return matches;
  }

  return { buildTimeline: buildTimeline, linkClaims: linkClaims, normText: normText };
});
