// Benchmark Schwab quotes latency (Node.js).
//
// Reads access_token from data/schwab_tokens.json and uses it directly.
// For fairness, run python scripts/bench_schwab_refresh.py first so the token is fresh.
//
// Usage:
//   node scripts/bench_schwab_quotes.js --symbols SPY,QQQ,TSLA --iters 20 --sleep-ms 3000

import fs from "node:fs";
import path from "node:path";

const argv = process.argv.slice(2);
const getArg = (k, def) => {
  const i = argv.indexOf(k);
  if (i === -1) return def;
  return argv[i + 1] ?? def;
};

const symbols = (getArg("--symbols", "SPY,QQQ,TSLA,AAPL,NVDA") || "")
  .split(",")
  .map((s) => s.trim().toUpperCase())
  .filter(Boolean);
const iters = parseInt(getArg("--iters", "20"), 10);
const sleepMs = parseInt(getArg("--sleep-ms", "3000"), 10);

const tokenPath = path.resolve("data", "schwab_tokens.json");
if (!fs.existsSync(tokenPath)) {
  console.error(`Missing token file: ${tokenPath}`);
  process.exit(2);
}

const tok = JSON.parse(fs.readFileSync(tokenPath, "utf-8"));
const accessToken = tok.access_token || tok.accessToken;
if (!accessToken) {
  console.error("No access_token in token file.");
  process.exit(2);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function pct(xs, p) {
  if (!xs.length) return null;
  const ys = [...xs].sort((a, b) => a - b);
  const k = (ys.length - 1) * p;
  const f = Math.floor(k);
  const c = Math.min(f + 1, ys.length - 1);
  if (f === c) return ys[f];
  return ys[f] * (c - k) + ys[c] * (k - f);
}

async function main() {
  const lats = [];
  let failures = 0;

  for (let i = 0; i < iters; i++) {
    const t0 = performance.now();
    let ok = true;
    try {
      const url = new URL("https://api.schwabapi.com/marketdata/v1/quotes");
      url.searchParams.set("symbols", symbols.join(","));

      const res = await fetch(url, {
        headers: {
          Authorization: `Bearer ${accessToken}`,
          Accept: "application/json",
        },
      });
      if (!res.ok) {
        ok = false;
      } else {
        // consume body
        await res.json().catch(() => null);
      }
    } catch {
      ok = false;
    }
    const t1 = performance.now();
    const dt = t1 - t0;
    if (ok) lats.push(dt);
    else failures++;
    await sleep(Math.max(0, sleepMs));
  }

  const out = {
    lang: "node",
    iters,
    ok: lats.length,
    fail: failures,
    symbols,
    lat_ms: {
      min: lats.length ? Math.min(...lats) : null,
      mean: lats.length ? lats.reduce((a, b) => a + b, 0) / lats.length : null,
      p50: pct(lats, 0.5),
      p95: pct(lats, 0.95),
      max: lats.length ? Math.max(...lats) : null,
    },
  };

  console.log(JSON.stringify(out, null, 2));
}

main();
