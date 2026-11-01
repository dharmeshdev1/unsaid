// The interface. Everything shown on the server's half of the page is computed from the bytes
// that really cross to the server. Nothing there is decoration.

const $ = (id) => document.getElementById(id);
const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;

const CELL = 7;
const BLOB_FILES = {
  routing: "routing.bin",
  layout: "layout.bin",
  page_digests: "page_digests.bin",
  rank_hint: "rank.hint",
  payload_hint: "payload.hint",
  model: "model.usm",
};
const KEEP_FOR_COMPARISON = 1 << 16;

function size(bytes) {
  if (bytes >= 1 << 20) return `${(bytes / (1 << 20)).toFixed(bytes >= 10 << 20 ? 0 : 1)} MB`;
  return `${Math.round(bytes / 1024)} KB`;
}

function ms(value) {
  return value >= 1000 ? `${(value / 1000).toFixed(1)} s` : `${Math.round(value)} ms`;
}

/** A worker that answers calls with promises. */
class Peer {
  constructor(onProgress) {
    this.worker = new Worker("engine.worker.js", { type: "module" });
    this.waiting = new Map();
    this.next = 0;
    this.worker.onmessage = ({ data }) => {
      if (data.progress !== undefined) {
        onProgress?.(data.progress);
        return;
      }
      const { resolve, reject } = this.waiting.get(data.id);
      this.waiting.delete(data.id);
      if (data.error) reject(new Error(data.error));
      else resolve(data.result);
    };
    this.worker.onerror = (event) => {
      for (const { reject } of this.waiting.values()) {
        reject(new Error(event.message || "the worker stopped"));
      }
      this.waiting.clear();
    };
  }

  call(op, args = {}, transfer = []) {
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.waiting.set(id, { resolve, reject });
      this.worker.postMessage({ id, op, args }, transfer);
    });
  }
}

/** Draws request bytes as marks: the slant and weight of each mark come from one real byte. */
class Pattern {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.marks = new Uint8Array(0);
    this.filled = 0; // Bytes accepted so far.
    this.drawn = 0; // Bytes on screen so far. Animation closes the gap between the two.
    this.frame = 0;
    this.tint = getComputedStyle(document.documentElement).getPropertyValue("--tint").trim();
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  resize() {
    const ratio = Math.min(devicePixelRatio || 1, 2);
    const { clientWidth: width, clientHeight: height } = this.canvas;
    if (!width || !height) return;
    this.canvas.width = Math.round(width * ratio);
    this.canvas.height = Math.round(height * ratio);
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.columns = Math.ceil(width / CELL);
    this.capacity = this.columns * Math.ceil(height / CELL);
    const kept = this.marks.subarray(0, Math.min(this.filled, this.capacity));
    this.marks = new Uint8Array(this.capacity);
    this.marks.set(kept);
    this.filled = kept.length;
    this.drawn = 0;
    this.drawUpTo(this.filled);
  }

  clear() {
    cancelAnimationFrame(this.frame);
    this.frame = 0;
    this.context.clearRect(0, 0, this.canvas.clientWidth, this.canvas.clientHeight);
    this.filled = 0;
    this.drawn = 0;
  }

  drawUpTo(target) {
    const c = this.context;
    c.strokeStyle = this.tint;
    c.lineWidth = 1.15;
    c.lineCap = "round";
    for (; this.drawn < target; this.drawn++) {
      const byte = this.marks[this.drawn];
      const x = (this.drawn % this.columns) * CELL;
      const y = Math.floor(this.drawn / this.columns) * CELL;
      c.globalAlpha = 0.45 + 0.55 * ((byte >> 1) & 3) / 3;
      c.beginPath();
      if (byte & 1) {
        c.moveTo(x + 0.5, y + 0.5);
        c.lineTo(x + CELL - 0.5, y + CELL - 0.5);
      } else {
        c.moveTo(x + CELL - 0.5, y + 0.5);
        c.lineTo(x + 0.5, y + CELL - 0.5);
      }
      c.stroke();
    }
  }

  animate() {
    if (this.frame) return;
    let last = performance.now();
    const perMs = this.capacity / 520; // The whole area takes about half a second.
    const step = (now) => {
      this.drawUpTo(Math.min(this.filled, this.drawn + Math.ceil((now - last) * perMs)));
      last = now;
      this.frame = this.drawn < this.filled ? requestAnimationFrame(step) : 0;
    };
    this.frame = requestAnimationFrame(step);
  }

  /** Add marks for these bytes, up to `share` of the whole area. Returns the bytes shown. */
  add(bytes, share) {
    const room = this.capacity - this.filled;
    const count = Math.min(bytes.length, Math.ceil(this.capacity * share), room);
    this.marks.set(bytes.subarray(0, count), this.filled);
    this.filled += count;
    if (calm) this.drawUpTo(this.filled);
    else this.animate();
    return count;
  }
}

/** The static demo site marks itself. Otherwise this page came from a real server. */
async function locate() {
  const demo = document.body.dataset.source === "demo";
  const response = await fetch(demo ? "demo/manifest.json" : "v1/manifest", { redirect: "error" });
  if (!response.ok) throw new Error("no search index is available on this page");
  const manifest = await response.text();
  return { live: !demo, manifest, parsed: JSON.parse(manifest) };
}

function createTransport(source, server) {
  if (server) {
    const send = async (op, request) => {
      const { answer, ms: spent } = await server.call(op, { request }, [request.buffer]);
      return { answer, ms: spent };
    };
    return { rank: (r) => send("rank", r), fetch: (r) => send("fetch", r) };
  }
  const send = async (path, request) => {
    const started = performance.now();
    const response = await fetch(path, {
      method: "POST",
      redirect: "error",
      headers: { "Content-Type": "application/octet-stream" },
      body: request,
    });
    if (!response.ok) throw new Error(`the server answered ${response.status}`);
    const answer = new Uint8Array(await response.arrayBuffer());
    return { answer, ms: performance.now() - started };
  };
  return { rank: (r) => send("v1/rank", r), fetch: (r) => send("v1/fetch", r) };
}

const state = {
  client: null,
  transport: null,
  info: null,
  ready: false,
  searching: false,
  queued: null,
  previous: null,
  pattern: null,
};

function setStatus(text, kind) {
  $("status").textContent = text;
  $("status").classList.toggle("error", kind === "error");
}

function setPhase(phase) {
  document.body.dataset.state = phase;
}

function showLogQuery(text) {
  $("log-query").textContent = encodeURIComponent(text.trim()).replaceAll("%20", "+");
}

function safeLink(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "https:" || parsed.protocol === "http:" ? parsed.href : null;
  } catch {
    return null;
  }
}

function renderHits(outcome, query) {
  const list = $("hits");
  list.replaceChildren();
  $("results").hidden = false;
  document.body.classList.add("has-results");
  for (const hit of outcome.hits) {
    const item = document.createElement("li");
    item.className = "hit";
    const title = document.createElement("h3");
    title.textContent = hit.title || "Untitled";
    const text = document.createElement("p");
    text.className = "passage short";
    text.textContent = hit.text;
    const more = document.createElement("button");
    more.type = "button";
    more.className = "more";
    more.textContent = "Show the whole passage";
    more.addEventListener("click", () => {
      text.classList.remove("short");
      more.remove();
    });
    item.append(title, text, more);
    const source = document.createElement("p");
    source.className = "source";
    const href = safeLink(hit.url);
    if (href) {
      const link = document.createElement("a");
      link.href = href;
      link.rel = "noopener noreferrer";
      link.target = "_blank";
      link.textContent = new URL(href).hostname.replace(/^www\./, "");
      source.append(link, `, similarity ${hit.score.toFixed(2)}`);
    } else {
      source.textContent = `Similarity ${hit.score.toFixed(2)}`;
    }
    item.append(source);
    list.append(item);
  }
  if (!outcome.hits.length) {
    const empty = document.createElement("li");
    empty.className = "hit";
    empty.textContent = `Nothing came back for "${query}". Try other words.`;
    list.append(empty);
  }
}

function sharedBytes(a, b) {
  const length = Math.min(a.length, b.length);
  let same = 0;
  for (let i = 0; i < length; i++) if (a[i] === b[i]) same++;
  return same / length;
}

function showLedger(trace) {
  $("ledger").hidden = false;
  $("led-received").textContent =
    `${size(trace.up[0])} and then ${size(trace.up[1])} of what looks like noise.`;
  $("led-did").textContent =
    `Multiplied them by its whole index, the same work for every search, in ${ms(trace.server)}.`;
  $("led-sent").textContent = `${size(trace.down)} it cannot read either.`;
  $("log-unsaid-code").textContent =
    `your-ip  POST /v1/rank   ${trace.up[0]} bytes\n` +
    `your-ip  POST /v1/fetch  ${trace.up[1]} bytes`;
  if (state.previous) {
    const share = sharedBytes(trace.head, state.previous);
    $("led-compare-row").hidden = false;
    $("led-compare").textContent =
      `Same sizes. ${(share * 100).toFixed(2)}% of bytes match, and chance alone gives 0.39%.`;
  }
  state.previous = trace.head;
}

async function search(query) {
  query = query.trim();
  if (!query) {
    $("query").focus();
    return;
  }
  showLogQuery(query);
  if (!state.ready) {
    state.queued = query;
    return;
  }
  if (state.searching) return;
  state.searching = true;
  setPhase("searching");
  $("go").disabled = true;
  setStatus("Encrypting your question on this device");
  const pattern = state.pattern;
  pattern.clear();
  $("pattern-empty").hidden = true;
  $("pattern-caption").textContent = "";

  try {
    const started = performance.now();
    const client = state.client;
    const begun = await client.call("begin", { query });
    const up0 = begun.request.length;
    const head = begun.request.slice(0, KEEP_FOR_COMPARISON);
    const shownRank = pattern.add(begun.request, 0.75);

    setStatus("The server is working on what it cannot read");
    const ranked = await state.transport.rank(begun.request);
    const down0 = ranked.answer.length;

    const scored = await client.call(
      "receiveRank",
      { answer: ranked.answer, limit: 10 },
      [ranked.answer.buffer],
    );
    const up1 = scored.request.length;
    const shownFetch = pattern.add(scored.request, 0.25);

    const fetched = await state.transport.fetch(scored.request);
    const down1 = fetched.answer.length;
    const opened = await client.call(
      "receivePages",
      { answer: fetched.answer },
      [fetched.answer.buffer],
    );
    const total = performance.now() - started;

    renderHits(opened.outcome, query);
    showLedger({
      up: [up0, up1],
      down: down0 + down1,
      server: ranked.ms + fetched.ms,
      head,
    });
    $("pattern-caption").textContent =
      `The first ${(shownRank + shownFetch).toLocaleString("en-US")} bytes of your search, ` +
      "exactly as the server received them. One mark per byte.";
    const decrypt = scored.decryptMs + opened.decryptMs;
    const server = ranked.ms + fetched.ms;
    const waited = total - begun.embedMs - begun.encryptMs - server - decrypt - scored.heldMs;
    const parts = `understanding the question took ${ms(begun.embedMs)}, ` +
      `encrypting ${ms(begun.encryptMs)}, the server ${ms(server)}, ` +
      `decrypting ${ms(decrypt)}` +
      (waited > 150 ? `, plus ${ms(waited)} waiting for encryption prepared ahead` : "");
    if (opened.outcome.corrupted_pages > 0) {
      setStatus(
        `${opened.outcome.corrupted_pages} pages failed their integrity check and were dropped. ` +
          "The server sent something other than its published index.",
        "error",
      );
    } else {
      const found = opened.outcome.hits.length;
      setStatus(`${found} results in ${ms(total)}: ${parts}.`);
    }
    setPhase("done");
  } catch (error) {
    state.client.call("abort").catch(() => {});
    setStatus(`The search failed: ${error.message}. Try again.`, "error");
    setPhase("error");
  } finally {
    state.searching = false;
    $("go").disabled = false;
    prepareAhead();
  }
}

let preparing = false;

/** Do the slow encryption work while the page is idle, so the next search starts instantly. */
async function prepareAhead() {
  if (preparing || !state.ready) return;
  preparing = true;
  try {
    for (;;) {
      const { ready, ms: spent } = await state.client.call("prepare");
      if (spent > 0) document.body.dataset.prepareMs = String(Math.round(spent));
      if (ready >= 2) break;
    }
  } catch {
    // A search prepares for itself if this did not happen.
  } finally {
    preparing = false;
  }
}

async function start() {
  const source = await locate();
  const manifest = source.parsed;
  const blobs = manifest.blobs;
  let total = Object.values(blobs).reduce((sum, blob) => sum + blob.bytes, 0);
  if (!source.live) {
    total += manifest.rank.rows * manifest.rank.cols;
    total += manifest.payload.rows * manifest.payload.cols;
  }

  const begin = async () => {
    $("start").hidden = true;
    $("meter").hidden = false;
    let loaded = 0;
    const onProgress = (delta) => {
      loaded += delta;
      $("meter-fill").style.width = `${Math.min(100, (loaded / total) * 100)}%`;
      setStatus(`Downloading the public index to this device, once: ${size(loaded)} of ${size(total)}`);
    };
    const client = new Peer(onProgress);
    const files = {};
    for (const [name, file] of Object.entries(BLOB_FILES)) {
      files[name] = {
        url: source.live ? `v1/blob/${blobs[name].sha256}` : `demo/${file}`,
        sha256: blobs[name].sha256,
        bytes: blobs[name].bytes,
      };
    }
    // A page from the server that holds the index already has to be trusted as much as the model.
    const loading = [client.call("loadClient", {
      manifest: source.manifest,
      files,
      allowUnknownModel: source.live,
    })];
    let server = null;
    if (!source.live) {
      server = new Peer(onProgress);
      loading.push(server.call("loadServer", {
        manifest: source.manifest,
        rank: {
          url: "demo/rank.matrix",
          key: `matrix/${blobs.rank_hint.sha256}`,
          rows: manifest.rank.rows,
          cols: manifest.rank.cols,
        },
        payload: {
          url: "demo/payload.matrix",
          key: `matrix/${blobs.payload_hint.sha256}`,
          rows: manifest.payload.rows,
          cols: manifest.payload.cols,
        },
      }));
    }
    const [info] = await Promise.all(loading);
    state.client = client;
    state.info = info;
    state.transport = createTransport(source, server);
    state.ready = true;
    $("meter").hidden = true;
    $("mode-note").textContent = source.live
      ? `Connected to a server holding ${info.documents.toLocaleString("en-US")} documents, ` +
        `index ${info.indexId}.`
      : `On this page the server is a separate browser worker, so the demo needs no backend. ` +
        `It receives only what a remote server would. It holds ` +
        `${info.documents.toLocaleString("en-US")} documents.`;
    setStatus(`Ready. ${info.documents.toLocaleString("en-US")} documents, and every search is encrypted.`);
    setPhase("ready");
    await prepareAhead();
    if (state.queued) {
      const query = state.queued;
      state.queued = null;
      await search(query);
    }
  };

  const savingData = navigator.connection?.saveData;
  if (savingData) {
    $("start").hidden = false;
    $("start").textContent = `Download the demo (${size(total)}, once)`;
    setStatus("This demo runs on your device, so it downloads its index first.");
    $("start").addEventListener("click", () => begin().catch(failLoad), { once: true });
  } else {
    await begin();
  }
}

function failLoad(error) {
  $("meter").hidden = true;
  setStatus(`The demo could not start: ${error.message}. Reload the page to try again.`, "error");
  setPhase("error");
}

function main() {
  state.pattern = new Pattern($("pattern"));
  const input = $("query");
  input.addEventListener("input", () => showLogQuery(input.value));
  $("form").addEventListener("submit", (event) => {
    event.preventDefault();
    search(input.value);
  });
  for (const button of document.querySelectorAll(".example")) {
    button.addEventListener("click", () => {
      input.value = button.textContent;
      search(input.value);
    });
  }
  showLogQuery(input.value);
  start().catch(failLoad);
}

main();
