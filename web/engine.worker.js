// Runs the Unsaid client, or on a page with no backend, the server. Each role gets its own
// worker, so on the demo page the two sides share no memory: everything between them is a
// message holding the same bytes a network would carry.

import init, { Engine, LocalServer } from "./pkg/unsaid_wasm.js";

const CACHE = "unsaid-v1";
let engine = null;
let server = null;
let wasmReady = null;

function wasm() {
  wasmReady ??= init();
  return wasmReady;
}

async function openCache() {
  try {
    return (await self.caches?.open(CACHE)) ?? null;
  } catch {
    return null; // Private windows may refuse storage. Everything still works without it.
  }
}

/** Download one file, reporting progress, and keep it for next time under `key`. */
async function download(url, key, size, report) {
  const cache = await openCache();
  const cached = await cache?.match(key).catch(() => null);
  if (cached) {
    const bytes = new Uint8Array(await cached.arrayBuffer());
    report(bytes.length);
    return bytes;
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} answered ${response.status}`);
  const bytes = new Uint8Array(size);
  const reader = response.body.getReader();
  let at = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (at + value.length > size) throw new Error(`${url} is larger than the index says`);
    bytes.set(value, at);
    at += value.length;
    report(value.length);
  }
  if (at !== size) throw new Error(`${url} is smaller than the index says`);
  await cache?.put(key, new Response(bytes)).catch(() => {});
  return bytes;
}

function progress(id) {
  return (delta) => self.postMessage({ id, progress: delta });
}

const handlers = {
  /** Fetch the public files and verify them. `files` maps each blob to its URL. */
  async loadClient({ manifest, files, allowUnknownModel }, id) {
    await wasm();
    const report = progress(id);
    const names = ["routing", "layout", "page_digests", "rank_hint", "payload_hint", "model"];
    const parts = await Promise.all(
      names.map((name) => {
        const { url, sha256, bytes } = files[name];
        return download(url, `blob/${sha256}`, bytes, report);
      }),
    );
    const text = new TextEncoder().encode(manifest);
    try {
      engine = new Engine(text, ...parts, allowUnknownModel);
    } catch (error) {
      await self.caches?.delete(CACHE).catch(() => {});
      throw error;
    }
    return {
      name: engine.name(),
      documents: engine.documents(),
      indexId: engine.index_id(),
      probes: engine.plan_probes(),
      pages: engine.plan_pages(),
    };
  },

  /** Fetch the matrices a real server would hold. Only the demo page does this. */
  async loadServer({ rank, payload }, id) {
    await wasm();
    const report = progress(id);
    const [rankBytes, payloadBytes] = await Promise.all([
      download(rank.url, rank.key, rank.rows * rank.cols, report),
      download(payload.url, payload.key, payload.rows * payload.cols, report),
    ]);
    server = new LocalServer(
      rank.rows, rank.cols, rankBytes, payload.rows, payload.cols, payloadBytes,
    );
    return {};
  },

  prepare() {
    const started = performance.now();
    if (engine.prepared_searches() < 2) engine.prepare_one();
    return { ms: performance.now() - started, ready: engine.prepared_searches() };
  },

  begin({ query }) {
    const t0 = performance.now();
    const embedding = engine.embed(query);
    const t1 = performance.now();
    const request = engine.begin(embedding);
    const t2 = performance.now();
    return { request, embedMs: t1 - t0, encryptMs: t2 - t1 };
  },

  async receiveRank({ answer, limit }) {
    const t0 = performance.now();
    engine.decrypt_rank(answer);
    const decrypting = performance.now() - t0;
    const request = engine.choose_pages(limit);
    const decryptMs = performance.now() - t0;
    // Ranking takes a time that depends on the results, and the next request is the one thing a
    // server can time. It is tiny next to the decryption, which never varies, so leaving at twice
    // that hides it.
    const held = Math.max(0, 2 * decrypting - decryptMs);
    if (held > 0) await new Promise((resolve) => setTimeout(resolve, held));
    return { request, decryptMs, heldMs: performance.now() - t0 - decryptMs };
  },

  abort() {
    engine?.abort();
    return {};
  },

  receivePages({ answer }) {
    const t0 = performance.now();
    const outcome = JSON.parse(engine.receive_pages(answer));
    return { outcome, decryptMs: performance.now() - t0 };
  },

  rank({ request }) {
    const t0 = performance.now();
    const answer = server.rank(request);
    return { answer, ms: performance.now() - t0 };
  },

  fetch({ request }) {
    const t0 = performance.now();
    const answer = server.fetch(request);
    return { answer, ms: performance.now() - t0 };
  },
};

self.onmessage = async ({ data: { id, op, args } }) => {
  try {
    const result = await handlers[op](args, id);
    const transfer = [result.request, result.answer]
      .filter(Boolean)
      .map((bytes) => bytes.buffer);
    self.postMessage({ id, result }, transfer);
  } catch (error) {
    self.postMessage({ id, error: error?.message ?? String(error) });
  }
};
