/* PaperFlow dashboard markdown renderer.

Extracted from index.html so it can be unit-tested with Node.
Works as a browser global (window.PaperFlowMD) and as a CommonJS module.

Supports: headings, bold, italic, links, inline code, code fences, lists,
blockquotes, paragraphs and pipe tables.
*/

(function (root, factory) {
  if (typeof module !== "undefined" && module.exports) {
    module.exports = factory();
  } else {
    root.PaperFlowMD = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function inline(t) {
    return t
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>")
      .replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  }

  function isSeparator(line) {
    return /^\s*\|?[\s:|-]+\|?\s*$/.test(line) && line.includes("-") && line.includes("|");
  }

  function splitRow(line) {
    return line.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|").map(function (c) { return c.trim(); });
  }

  function renderTable(rows) {
    var html = "<table>";
    rows.forEach(function (cells, i) {
      var tag = i === 0 ? "th" : "td";
      html += "<tr>" + cells.map(function (c) { return "<" + tag + ">" + inline(c) + "</" + tag + ">"; }).join("") + "</tr>";
    });
    return html + "</table>";
  }

  function md(src) {
    var lines = esc(src).split("\n");
    var html = "", inCode = false, inList = false;
    for (var i = 0; i < lines.length; i++) {
      var raw = lines[i];
      var line = raw.trimEnd();
      if (line.trim().startsWith("```")) {
        inCode = !inCode;
        html += inCode ? "<pre><code>" : "</code></pre>\n";
        continue;
      }
      if (inCode) { html += line + "\n"; continue; }
      /* pipe table: a header row followed by a separator row */
      if (line.includes("|") && i + 1 < lines.length && isSeparator(lines[i + 1].trimEnd())) {
        var rows = [splitRow(line)];
        var j = i + 2;
        while (j < lines.length && lines[j].trim().startsWith("|") && lines[j].includes("|")) {
          rows.push(splitRow(lines[j]));
          j++;
        }
        html += renderTable(rows);
        i = j - 1;
        continue;
      }
      var m = line.match(/^(#{1,3})\s+(.*)$/);
      if (m) { var lvl = m[1].length; html += "<h" + lvl + ">" + inline(m[2]) + "</h" + lvl + ">"; continue; }
      if (line.startsWith("> ")) { html += "<blockquote>" + inline(line.slice(2)) + "</blockquote>"; continue; }
      if (/^\s*[-*]\s+/.test(line)) {
        if (!inList) { html += "<ul>"; inList = true; }
        html += "<li>" + inline(line.replace(/^\s*[-*]\s+/, "")) + "</li>";
        continue;
      }
      if (inList) { html += "</ul>"; inList = false; }
      if (line.trim() === "") { html += "\n"; continue; }
      html += "<p>" + inline(line) + "</p>";
    }
    if (inList) html += "</ul>";
    return html;
  }

  return { esc: esc, md: md };
});
