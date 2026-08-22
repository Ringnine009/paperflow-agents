/* Node unit tests for the dashboard markdown renderer (tables, escaping). */

import { createRequire } from "module";
import assert from "node:assert";

const require = createRequire(import.meta.url);
const { md } = require("../paperflow/web/static/markdown.js");

const html = md(
  "| Metric | Value |\n" +
  "|---|---|\n" +
  "| Win rate | 68.8% |\n" +
  "| Vote accuracy | 66.6% |\n"
);
assert(html.includes("<table>"), "opens <table>");
assert(html.includes("<th>Metric</th>"), "header cell");
assert(html.includes("<th>Value</th>"), "second header cell");
assert(html.includes("<td>68.8%</td>"), "body cell");
assert(html.includes("</table>"), "closes </table>");

/* HTML in table cells must stay escaped */
const safe = md("| a | b |\n|---|---|\n| <script>alert(1)</script> | &amp; |\n");
assert(!safe.includes("<script>"), "table cells are escaped");
assert(safe.includes("&amp;"), "ampersand preserved");

/* a lone pipe is not a table */
const noTable = md("just some a | b text");
assert(!noTable.includes("<table>"), "single pipe line is not a table");

/* plain markdown still works */
const plain = md("# Title\n\nSome **bold** text and [a link](https://example.com).\n");
assert(plain.includes("<h1>Title</h1>"), "heading");
assert(plain.includes("<strong>bold</strong>"), "bold");
assert(plain.includes('href="https://example.com"'), "link");

console.log("markdown renderer tests OK");
