// Headless test runner for the `gpu` area.
//   node library/tests/gpu/run.mjs [id ...]        (default: every library/tests/gpu/*.test.js)
// Serves library/ on a free localhost port, opens tests/gpu/runner.html?test=<id> in Chromium with WebGPU
// (software adapter SwiftShader unless GPU_ADAPTER is set), prints every row, exits 1 on any failure,
// console error, page error, missing row or timeout.
//   CHROMIUM=/path/to/chrome   browser executable (default /opt/pw-browsers/chromium, else Playwright's own)
//   GPU_ADAPTER=swiftshader|default
import { createRequire } from "node:module";
import { execSync } from "node:child_process";
import { createServer } from "node:http";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { join, dirname, extname, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = normalize(join(here, "..", ".."));            // library/
const require = createRequire(import.meta.url);
let playwright;
try { playwright = require("playwright"); }
catch { playwright = require(execSync("npm root -g").toString().trim() + "/playwright"); }

const ids = process.argv.slice(2).length ? process.argv.slice(2)
  : readdirSync(here).filter(f => f.endsWith(".test.js")).map(f => f.slice(0, -8)).sort();
if (!ids.length) { console.error("no GPU tests found"); process.exit(1); }

const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
  ".wgsl": "text/plain" };
const server = createServer((req, res) => {
  const p = normalize(join(root, decodeURIComponent(new URL(req.url, "http://x").pathname)));
  if (!p.startsWith(root) || !existsSync(p)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { "content-type": TYPES[extname(p)] || "application/octet-stream" });
  res.end(readFileSync(p));
});
await new Promise(r => server.listen(0, "127.0.0.1", r));
const port = server.address().port;

const exe = process.env.CHROMIUM || (existsSync("/opt/pw-browsers/chromium") ? "/opt/pw-browsers/chromium" : undefined);
const adapter = process.env.GPU_ADAPTER || "swiftshader";
const args = ["--enable-unsafe-webgpu", "--no-sandbox", ...(adapter === "default" ? [] : [`--use-webgpu-adapter=${adapter}`])];
const browser = await playwright.chromium.launch({ executablePath: exe, args });

let failed = 0;
for (const id of ids) {
  const page = await browser.newPage();
  const errs = [];
  page.on("console", m => { if (m.type() === "error") errs.push(m.text()); });
  page.on("pageerror", e => errs.push(String(e)));
  const t0 = Date.now();
  let R;
  try {
    await page.goto(`http://localhost:${port}/tests/gpu/runner.html?test=${id}`);
    await page.waitForFunction(() => window.RESULT, null, { timeout: 300000 });
    R = await page.evaluate(() => window.RESULT);
  } catch (e) { R = { error: String(e) }; }
  await page.close();
  const secs = ((Date.now() - t0) / 1000).toFixed(1);
  if (R.error) { failed++; console.log(`FAIL ${id}: ${R.error}`); }
  else {
    for (const row of R.rows) {
      if (!row.pass) failed++;
      const e = typeof row.err === "number" ? row.err.toExponential(2) : String(row.err);
      const t = typeof row.tol === "number" ? row.tol.toExponential(0) : String(row.tol);
      console.log(`${row.pass ? "ok  " : "FAIL"} ${id}: ${row.name.padEnd(64)} err ${e.padStart(9)}  tol ${t}`);
    }
  }
  if (errs.length) { failed++; console.log(`FAIL ${id}: console/page errors: ${errs.join(" | ")}`); }
  console.log(`---- ${id} (${secs} s)`);
}
await browser.close();
server.close();
console.log(failed ? `${failed} GPU check(s) failed` : `all GPU checks passed (${ids.length} routines, adapter ${adapter})`);
process.exit(failed ? 1 : 0);
