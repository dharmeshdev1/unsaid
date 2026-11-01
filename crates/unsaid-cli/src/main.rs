//! Search without saying what you're looking for.

mod assets;
mod engine;
mod eval;
mod mcp;
mod models;
mod proxy;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use clap::{Args, Parser, Subcommand};
use rayon::prelude::*;
use unsaid_build::{BuildOptions, IngestOptions, build, read_documents};
use unsaid_core::client::SearchPlan;
use unsaid_core::manifest::{index_id, sha256_hex};
use unsaid_embed::Embedder;
use unsaid_server::Loaded;

use crate::engine::{Engine, Trace};

#[derive(Parser)]
#[command(
    name = "unsaid",
    version,
    about = "Search without saying what you're looking for.",
    long_about = "A search engine that cannot see what you search for. Queries are encrypted on \
                  your own device with lattice cryptography, and the server finds relevant \
                  documents and returns them without learning the query or the results."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build a private search index from documents.
    Build(BuildArgs),
    /// Serve an index and the web interface.
    Serve(ServeArgs),
    /// Search privately from the terminal.
    Search(SearchArgs),
    /// Try it right now: index a small sample and serve a private search page.
    Demo(DemoArgs),
    /// Run a local HTTP gateway so any program can search privately.
    Proxy(ProxyArgs),
    /// Give an AI agent a private search tool over the Model Context Protocol.
    Mcp(ConnectArgs),
    /// Show an index's parameters, costs and security margins.
    Info(InfoArgs),
    /// Measure search latency and traffic.
    Bench(BenchArgs),
    /// Measure retrieval quality on a BEIR dataset through the real protocol.
    Eval(EvalArgs),
    /// Print a query's embedding as JSON, computed on this machine.
    Embed(EmbedArgs),
}

#[derive(Args, Clone)]
struct IndexTuning {
    /// Embedding model: minilm, bge, or a path to a packed model.
    #[arg(long, default_value = "minilm")]
    model: String,
    /// Vector length after projection. Higher is more accurate and costs more traffic.
    #[arg(long, default_value_t = 256)]
    dim: usize,
    /// Most documents per cluster. Chosen from the corpus size if not given.
    #[arg(long)]
    rows: Option<usize>,
    /// Bytes per document page. Chosen from the corpus size if not given.
    #[arg(long)]
    page_bytes: Option<usize>,
    /// Clusters scored per search by default.
    #[arg(long, default_value_t = 8)]
    probes: usize,
    /// Pages retrieved per search by default.
    #[arg(long, default_value_t = 4)]
    pages: usize,
    /// Routing prototypes per cluster.
    #[arg(long, default_value_t = 32)]
    prototypes: usize,
    /// Longest text kept per document, in bytes.
    #[arg(long, default_value_t = 600)]
    max_text_bytes: usize,
    /// Longest input to the embedding model, in tokens.
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,
}

impl IndexTuning {
    fn options(&self, name: String) -> BuildOptions {
        let mut o = BuildOptions::new(name);
        o.dim = self.dim;
        o.rows = self.rows;
        o.page_bytes = self.page_bytes;
        o.probes = self.probes;
        o.pages = self.pages;
        o.prototypes = self.prototypes;
        o.max_text_bytes = self.max_text_bytes;
        o
    }
}

#[derive(Args)]
struct BuildArgs {
    /// A folder of text, Markdown and HTML files, or a .jsonl file of {"title","text","url"}.
    source: PathBuf,
    /// Output directory.
    #[arg(short, long, default_value = "index")]
    out: PathBuf,
    /// Name shown to searchers.
    #[arg(long)]
    name: Option<String>,
    /// Longest passage, in words.
    #[arg(long, default_value_t = 120)]
    chunk_words: usize,
    /// Reuse or create a file of raw embeddings, so a rebuild need not embed again.
    #[arg(long)]
    embeddings: Option<PathBuf>,
    #[command(flatten)]
    tuning: IndexTuning,
}

#[derive(Args)]
struct ServeArgs {
    /// Index directory.
    #[arg(default_value = "index")]
    index: PathBuf,
    /// Address to listen on. Use 0.0.0.0 to accept connections from other machines.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// Port to listen on.
    #[arg(short, long, default_value_t = 8377)]
    port: u16,
    /// Print every query as it arrives, which shows all that the server can see of it.
    #[arg(long)]
    watch: bool,
}

#[derive(Args, Clone)]
struct ConnectArgs {
    /// Server URL.
    #[arg(long, short, default_value = "http://127.0.0.1:8377")]
    server: String,
    /// Refuse to connect unless the server's index has this identifier.
    #[arg(long)]
    pin: Option<String>,
    /// Clusters scored per search. More is more thorough and sends more.
    #[arg(long)]
    probes: Option<usize>,
    /// Pages retrieved per search.
    #[arg(long)]
    pages: Option<usize>,
    /// Accept an embedding model with this SHA-256, beyond the ones this client knows. A model
    /// shapes how your query is handled here, so only use one you have checked.
    #[arg(long = "allow-model", value_name = "SHA256")]
    allowed_models: Vec<String>,
}

impl ConnectArgs {
    fn plan(&self, engine: &Engine) -> SearchPlan {
        let default = engine.index().default_plan();
        SearchPlan {
            probes: self.probes.unwrap_or(default.probes).clamp(1, 64),
            pages: self.pages.unwrap_or(default.pages).clamp(1, 64),
        }
    }
}

#[derive(Args)]
struct SearchArgs {
    /// What to look for.
    query: Vec<String>,
    /// Results to show.
    #[arg(short = 'k', long, default_value_t = 5)]
    limit: usize,
    /// Print JSON.
    #[arg(long)]
    json: bool,
    /// Also show what was sent and how long each step took.
    #[arg(long)]
    trace: bool,
    /// Search a local index directory instead of a server.
    #[arg(long)]
    index: Option<PathBuf>,
    #[command(flatten)]
    connect: ConnectArgs,
}

#[derive(Args)]
struct DemoArgs {
    /// Port to listen on.
    #[arg(short, long, default_value_t = 8377)]
    port: u16,
    /// Embedding model.
    #[arg(long, default_value = "minilm")]
    model: String,
    /// Print every query as it arrives, which shows all that the server can see of it.
    #[arg(long)]
    watch: bool,
}

#[derive(Args)]
struct ProxyArgs {
    /// Address for the local gateway.
    #[arg(long, default_value = "127.0.0.1:8378")]
    listen: String,
    #[command(flatten)]
    connect: ConnectArgs,
}

#[derive(Args)]
struct InfoArgs {
    /// Index directory.
    #[arg(default_value = "index")]
    index: PathBuf,
}

#[derive(Args)]
struct BenchArgs {
    /// Number of searches.
    #[arg(short, long, default_value_t = 100)]
    n: usize,
    /// Use a local index directory, with the server's arithmetic in this process.
    #[arg(long)]
    index: Option<PathBuf>,
    /// A file of queries, one per line. Built-in queries are used if not given.
    #[arg(long)]
    queries: Option<PathBuf>,
    /// Print JSON.
    #[arg(long)]
    json: bool,
    #[command(flatten)]
    connect: ConnectArgs,
}

#[derive(Args)]
struct EmbedArgs {
    /// The text to embed.
    text: Vec<String>,
    /// Embedding model: minilm, bge, or a path to a packed model.
    #[arg(long, default_value = "minilm")]
    model: String,
}

#[derive(Args)]
struct EvalArgs {
    /// A BEIR dataset directory.
    dataset: PathBuf,
    /// Where to write the index and cached embeddings.
    #[arg(short, long, default_value = "eval-index")]
    out: PathBuf,
    /// Probe counts to compare.
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,16")]
    probe_counts: Vec<usize>,
    /// Page counts to compare.
    #[arg(long, value_delimiter = ',', default_value = "4,8")]
    page_counts: Vec<usize>,
    /// Evaluate at most this many queries, for very large datasets.
    #[arg(long)]
    max_queries: Option<usize>,
    #[command(flatten)]
    tuning: IndexTuning,
}

/// Read previously computed embeddings, if the file exists and has the right size.
pub(crate) fn cached_embeddings(path: &Path, rows: usize, dim: usize) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() != rows * dim * 4 {
        return None;
    }
    eprintln!("using embeddings from {}", path.display());
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// Embed texts on every core, optionally keeping the result in a file.
pub(crate) fn embed_corpus(
    embedder: &Embedder,
    texts: &[String],
    max_tokens: usize,
    cache: Option<&Path>,
) -> Result<Vec<f32>, String> {
    if let Some(found) = cache.and_then(|p| cached_embeddings(p, texts.len(), embedder.dim())) {
        return Ok(found);
    }
    let total = texts.len();
    let done = AtomicUsize::new(0);
    let started = Instant::now();
    let rows: Vec<Vec<f32>> = texts
        .par_iter()
        .map(|text| {
            let row = embedder.embed_document(text, max_tokens);
            let so_far = done.fetch_add(1, Ordering::Relaxed) + 1;
            if so_far % 5000 == 0 || so_far == total {
                let rate = so_far as f64 / started.elapsed().as_secs_f64().max(1e-3);
                eprintln!("  embedded {so_far}/{total} ({rate:.0} per second)");
            }
            row
        })
        .collect();
    let flat: Vec<f32> = rows.into_iter().flatten().collect();
    if let Some(path) = cache {
        let bytes: Vec<u8> = flat.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(flat)
}

fn megabytes(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.2} GB", bytes as f64 / (1u64 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.1} MB", bytes as f64 / (1u64 << 20) as f64)
    } else {
        format!("{:.0} KB", bytes as f64 / 1024.0)
    }
}

fn run_build(args: &BuildArgs) -> Result<(), String> {
    let ingest = IngestOptions {
        chunk_words: args.chunk_words,
        ..IngestOptions::default()
    };
    let documents = read_documents(&args.source, &ingest)?;
    let name = args.name.clone().unwrap_or_else(|| {
        args.source.file_stem().map_or_else(
            || "Unsaid index".to_owned(),
            |s| s.to_string_lossy().into_owned(),
        )
    });
    eprintln!(
        "{} passages from {}",
        documents.len(),
        args.source.display()
    );
    let model = models::resolve(&args.tuning.model)?;
    let embedder = Embedder::load(&model).map_err(|e| e.to_string())?;
    let texts: Vec<String> = documents
        .iter()
        .map(|d| {
            if d.title.is_empty() {
                d.text.clone()
            } else {
                format!("{}. {}", d.title, d.text)
            }
        })
        .collect();
    let embeddings = embed_corpus(
        &embedder,
        &texts,
        args.tuning.max_tokens,
        args.embeddings.as_deref(),
    )?;
    eprintln!("building the index");
    let started = Instant::now();
    let report = build(
        &documents,
        &embeddings,
        embedder.name(),
        embedder.dim(),
        &model,
        &args.tuning.options(name),
        &args.out,
    )?;
    eprintln!("done in {:.1} s\n", started.elapsed().as_secs_f64());
    println!("index {}", report.index_id);
    println!("  documents         {}", report.documents);
    println!(
        "  clusters          {} of up to {} documents",
        report.clusters, report.rank_rows
    );
    println!(
        "  pages             {} of {} bytes",
        report.pages, report.page_bytes
    );
    println!("  one-time download {}", megabytes(report.public_bytes));
    println!(
        "  per search        {} up, {} down",
        megabytes(report.upload_per_search),
        megabytes(report.download_per_search)
    );
    println!("  server memory     {}", megabytes(report.private_bytes));
    println!("\nStart it with: unsaid serve {}", args.out.display());
    let model_hash = sha256_hex(&model);
    if !unsaid_embed::is_known_model(&model_hash) {
        println!("This model is not on the trusted list, so clients must opt in to it with:");
        println!("  --allow-model {model_hash}");
    }
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

fn serve(index: &Path, host: &str, port: u16, watch: bool) -> Result<(), String> {
    let loaded = Arc::new(Loaded::open(index)?);
    let manifest = loaded.manifest();
    println!(
        "{}: {} documents, index {}",
        manifest.name,
        manifest.documents,
        loaded.index_id()
    );
    let address = format!("{host}:{port}");
    let shown = if host == "0.0.0.0" {
        format!("127.0.0.1:{port}")
    } else {
        address.clone()
    };
    println!("Private search at http://{shown}");
    println!("This server never sees what is searched for. Ctrl-C to stop.");
    if watch {
        println!("Printing every query exactly as it arrives:\n");
    }
    let runtime = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&address)
            .await
            .map_err(|e| format!("{address}: {e}"))?;
        let app = unsaid_server::router(loaded, assets::ASSETS, watch);
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| e.to_string())
    })
}

fn print_trace(trace: &Trace) {
    eprintln!(
        "\nsent {} and received {}. embed {:.0} ms, encrypt {:.0} ms, ranking {:.0} ms, pages {:.0} ms, decrypt {:.1} ms, held for timing {:.0} ms, total {:.0} ms",
        megabytes(trace.uploaded),
        megabytes(trace.downloaded),
        trace.embed_ms,
        trace.encrypt_ms,
        trace.rank_ms,
        trace.fetch_ms,
        trace.decrypt_ms,
        trace.held_ms,
        trace.total_ms(),
    );
}

fn connect(index: Option<&Path>, args: &ConnectArgs) -> Result<Engine, String> {
    match index {
        Some(dir) => Engine::local(dir),
        None => Engine::connect(&args.server, args.pin.as_deref(), &args.allowed_models),
    }
}

/// Text from a server is going to a terminal, where escape sequences can rewrite the screen, hide
/// a warning, or set the clipboard. Text direction overrides can disguise what is shown.
fn printable(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}' => ' ',
            c if c.is_control() => ' ',
            c => c,
        })
        .collect()
}

fn run_search(args: &SearchArgs) -> Result<(), String> {
    let query = args.query.join(" ");
    if query.trim().is_empty() {
        return Err(
            "say what to search for, for example: unsaid search \"how vaccines work\"".into(),
        );
    }
    let engine = connect(args.index.as_deref(), &args.connect)?;
    let plan = args.connect.plan(&engine);
    let (mut outcome, trace) = engine.search(&query, args.limit, plan)?;
    for hit in &mut outcome.hits {
        hit.document.title = printable(&hit.document.title);
        hit.document.text = printable(&hit.document.text);
        hit.document.url = printable(&hit.document.url);
    }
    if args.json {
        let value = serde_json::json!({"outcome": outcome, "trace": trace});
        println!("{value}");
        return Ok(());
    }
    if outcome.hits.is_empty() {
        println!("No results.");
    }
    for (i, hit) in outcome.hits.iter().enumerate() {
        println!("{}. {}  ({:.2})", i + 1, hit.document.title, hit.score);
        let snippet: String = hit.document.text.chars().take(240).collect();
        println!("   {snippet}");
        if !hit.document.url.is_empty() {
            println!("   {}", hit.document.url);
        }
    }
    if outcome.corrupted_pages > 0 {
        eprintln!(
            "\nwarning: {} pages failed their integrity check and were dropped",
            outcome.corrupted_pages
        );
    }
    if args.trace {
        print_trace(&trace);
    }
    Ok(())
}

const SAMPLE_CORPUS: &str = include_str!("../../../examples/sample.jsonl");

fn run_demo(args: &DemoArgs) -> Result<(), String> {
    // The version is in the name, so a newer format never finds an older index here.
    let dir = models::cache_dir().join(format!(
        "demo-index-{}-{}",
        env!("CARGO_PKG_VERSION"),
        args.model.replace('/', "_")
    ));
    if !dir.join("manifest.json").is_file() {
        eprintln!("First run: indexing 1,000 sample articles, which takes under a minute.");
        let source = models::cache_dir().join("demo-sample.jsonl");
        std::fs::create_dir_all(models::cache_dir()).map_err(|e| e.to_string())?;
        std::fs::write(&source, SAMPLE_CORPUS).map_err(|e| e.to_string())?;
        let tuning = IndexTuning {
            model: args.model.clone(),
            dim: 256,
            rows: None,
            page_bytes: None,
            probes: 8,
            pages: 4,
            prototypes: 32,
            max_text_bytes: 600,
            max_tokens: 128,
        };
        run_build(&BuildArgs {
            source,
            out: dir.clone(),
            name: Some("Sample of Simple English Wikipedia".into()),
            chunk_words: 200,
            embeddings: None,
            tuning,
        })?;
    }
    serve(&dir, "127.0.0.1", args.port, args.watch)
}

fn run_info(args: &InfoArgs) -> Result<(), String> {
    let path = args.index.join("manifest.json");
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let m = unsaid_core::Manifest::parse(&bytes).map_err(|e| e.to_string())?;
    println!("{}", m.name);
    println!("  index id          {}", index_id(&bytes));
    println!("  documents         {}", m.documents);
    println!(
        "  model             {} ({} dimensions, projected to {})",
        m.model, m.model_dim, m.dim
    );
    println!(
        "  lattice           n = {}, q = 2^{}, sigma = {} (compiled into every client)",
        m.lwe.n, m.lwe.log_q, m.lwe.sigma
    );
    println!(
        "  ranking matrix    {} x {}, plaintext 2^{}, failure bound 2^{:.0} per answer",
        m.rank.rows, m.rank.cols, m.rank.log_p, m.rank.failure_log2
    );
    println!(
        "  document matrix   {} x {}, failure bound 2^{:.0} per answer",
        m.payload.rows, m.payload.cols, m.payload.failure_log2
    );
    let public: u64 = m.blobs.values().map(|b| b.bytes).sum();
    println!("  one-time download {}", megabytes(public));
    let up = (m.probes * m.rank.cols + m.pages * m.payload.cols) as u64 * 4;
    let down = (m.probes * m.rank.rows + m.pages * m.payload.rows) as u64 * 4;
    println!(
        "  per search        {} up, {} down ({} probes, {} pages)",
        megabytes(up),
        megabytes(down),
        m.probes,
        m.pages
    );
    Ok(())
}

const BENCH_QUERIES: &[&str] = &[
    "what are the early signs of depression",
    "how do vaccines train the immune system",
    "why is the sky blue",
    "causes of the first world war",
    "how does a bill become law",
    "what do black holes do to light",
    "treatment for high blood pressure",
    "who painted the mona lisa",
    "how do bees make honey",
    "what happens during a solar eclipse",
];

fn run_bench(args: &BenchArgs) -> Result<(), String> {
    let engine = connect(args.index.as_deref(), &args.connect)?;
    let plan = args.connect.plan(&engine);
    let owned: Vec<String> = match &args.queries {
        Some(path) => std::fs::read_to_string(path)
            .map_err(|e| format!("{}: {e}", path.display()))?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(str::to_owned)
            .collect(),
        None => BENCH_QUERIES.iter().map(|s| (*s).to_owned()).collect(),
    };
    if owned.is_empty() {
        return Err("no queries to run".into());
    }
    let mut traces = Vec::with_capacity(args.n);
    for i in 0..args.n {
        let (outcome, trace) = engine.search(&owned[i % owned.len()], 10, plan)?;
        if outcome.corrupted_pages > 0 {
            return Err(format!("{} pages were corrupted", outcome.corrupted_pages));
        }
        traces.push(trace);
    }
    let stat = |f: &dyn Fn(&Trace) -> f64| {
        let mut values: Vec<f64> = traces.iter().map(f).collect();
        values.sort_by(f64::total_cmp);
        let at = |q: f64| values[((q * values.len() as f64) as usize).min(values.len() - 1)];
        (at(0.5), at(0.95))
    };
    let rows: [(&str, (f64, f64)); 7] = [
        ("embed query", stat(&|t| t.embed_ms)),
        ("encrypt", stat(&|t| t.encrypt_ms)),
        ("server: ranking", stat(&|t| t.rank_ms)),
        ("server: pages", stat(&|t| t.fetch_ms)),
        ("decrypt", stat(&|t| t.decrypt_ms)),
        ("held for timing", stat(&|t| t.held_ms)),
        ("total", stat(&|t| t.total_ms())),
    ];
    let m = engine.index().manifest();
    if args.json {
        let table: serde_json::Map<String, serde_json::Value> = rows
            .iter()
            .map(|(k, (a, b))| {
                (
                    (*k).to_owned(),
                    serde_json::json!({"p50_ms": a, "p95_ms": b}),
                )
            })
            .collect();
        println!(
            "{}",
            serde_json::json!({
                "documents": m.documents, "probes": plan.probes, "pages": plan.pages,
                "uploaded": traces[0].uploaded, "downloaded": traces[0].downloaded,
                "times": table,
            })
        );
        return Ok(());
    }
    println!(
        "{} searches over {} documents, {} probes and {} pages per search",
        args.n, m.documents, plan.probes, plan.pages
    );
    println!(
        "{} up, {} down per search\n",
        megabytes(traces[0].uploaded),
        megabytes(traces[0].downloaded)
    );
    println!("{:<18}{:>10}{:>10}", "step", "p50 ms", "p95 ms");
    for (name, (p50, p95)) in rows {
        println!("{name:<18}{p50:>10.1}{p95:>10.1}");
    }
    Ok(())
}

fn run() -> Result<(), String> {
    match Cli::parse().command {
        Command::Build(args) => run_build(&args),
        Command::Serve(args) => serve(&args.index, &args.host, args.port, args.watch),
        Command::Search(args) => run_search(&args),
        Command::Demo(args) => run_demo(&args),
        Command::Proxy(args) => {
            let engine = connect(None, &args.connect)?;
            let plan = args.connect.plan(&engine);
            let runtime = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
            runtime.block_on(proxy::run(engine, plan, &args.listen))
        }
        Command::Mcp(args) => {
            let engine = Arc::new(connect(None, &args)?);
            let plan = args.plan(&engine);
            engine.prepare_in_background(plan, 2);
            mcp::run(&engine, plan)
        }
        Command::Embed(args) => {
            let model = models::resolve(&args.model)?;
            let embedder = Embedder::load(&model).map_err(|e| e.to_string())?;
            let vector = embedder.embed_query(&args.text.join(" "));
            println!(
                "{}",
                serde_json::to_string(&vector).map_err(|e| e.to_string())?
            );
            Ok(())
        }
        Command::Info(args) => run_info(&args),
        Command::Bench(args) => run_bench(&args),
        Command::Eval(args) => eval::run(&eval::EvalOptions {
            dataset: args.dataset,
            out: args.out,
            model: models::resolve(&args.tuning.model)?,
            build: args.tuning.options("evaluation".into()),
            probes: args.probe_counts,
            pages: args.page_counts,
            max_tokens: args.tuning.max_tokens,
            max_queries: args.max_queries,
        }),
    }
}

fn main() {
    if let Err(message) = run() {
        // An error can quote text from a server.
        eprintln!("error: {}", printable(&message));
        std::process::exit(1);
    }
}
