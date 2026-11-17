// Render a BIRD-Deck HTML file: one PNG per slide + JSON overflow report.
// Usage: node scripts/deck_check.mjs deck.html out_dir [--scale 1|2]
// Exit code 0 = every slide fits, 1 = at least one problem (used as a hard reward / data filter).
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";

// CommonJS resolution so a globally pre-installed Playwright (NODE_PATH) is found too.
const { chromium } = createRequire(import.meta.url)("playwright");

const [html, outDir] = process.argv.slice(2);
const si = process.argv.indexOf("--scale");
const scale = si > 0 ? Number(process.argv[si + 1]) : 1;
if (!html || !outDir) {
  console.error("usage: deck_check.mjs deck.html out_dir [--scale N]");
  process.exit(2);
}
fs.mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: scale });
await page.goto("file://" + path.resolve(html));
await page.evaluate(() => document.fonts.ready);

const report = await page.evaluate(() => {
  const out = [];
  document.querySelectorAll("section.slide").forEach((slide) => {
    const sb = slide.getBoundingClientRect();
    const problems = [];
    // 1) clipped containers: content taller/wider than its box
    slide.querySelectorAll(".fit, h2, .card, .kpi, .takeaway, td").forEach((el) => {
      if (el.scrollHeight > el.clientHeight + 1 || el.scrollWidth > el.clientWidth + 1)
        problems.push({ kind: "clipped", el: String(el.className || el.tagName), over: [el.scrollWidth - el.clientWidth, el.scrollHeight - el.clientHeight] });
    });
    // 2) text painted outside the slide, or colliding with the footer
    const footer = slide.querySelector("footer");
    const fb = footer ? footer.getBoundingClientRect() : null;
    slide.querySelectorAll("*").forEach((el) => {
      if (el.closest("footer") || el.closest("svg")) return;
      const b = el.getBoundingClientRect();
      if (!b.width || !b.height) return;
      if (b.right > sb.right + 1 || b.bottom > sb.bottom + 1 || b.left < sb.left - 1)
        problems.push({ kind: "outside", el: String(el.className || el.tagName) });
      else if (fb && el.childElementCount === 0 && el.textContent.trim() && b.bottom > fb.top - 4)
        problems.push({ kind: "footer_overlap", el: String(el.className || el.tagName) });
    });
    out.push({ index: Number(slide.dataset.index), layout: slide.className.split(" ")[1].slice(2), problems });
  });
  return out;
});

const slides = await page.$$("section.slide");
for (let i = 0; i < slides.length; i++)
  await slides[i].screenshot({ path: path.join(outDir, `slide-${String(i + 1).padStart(2, "0")}.png`) });
await browser.close();

const result = { slides: report.length, ok: report.every((s) => s.problems.length === 0), report };
fs.writeFileSync(path.join(outDir, "report.json"), JSON.stringify(result, null, 2));
const bad = report.filter((s) => s.problems.length).map((s) => ({ i: s.index, layout: s.layout, problems: s.problems.slice(0, 3) }));
console.log(JSON.stringify({ slides: result.slides, ok: result.ok, bad }, null, 1));
process.exit(result.ok ? 0 : 1);
