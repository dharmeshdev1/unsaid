# Benchmarks

Every number here comes from a command you can run yourself. Measurements were taken on an Apple M5
with 10 cores and 16 GB of memory, with client and server on one machine, so **no network time is
included**. Add your own: at 20 Mbit/s up, a 2.3 MB query takes about a second to send.

## Search quality

`unsaid eval` builds the index and sends every query through the real protocol with real encryption.
The baseline is exact search over the original embeddings from the same model, all-MiniLM-L6-v2. So 100%
means that clustering, projection and encryption lost nothing. It says nothing about how good the
model is, and a better model would lift every row.

These use the defaults of 8 probes and 4 pages, and are from the [BEIR](https://github.com/beir-cellar/beir)
benchmark:

| Dataset | Documents | Queries | Exact nDCG@10 | Unsaid nDCG@10 | Share of exact | Up | Down |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SciFact | 5,183 | 300 | 0.6230 | 0.5969 | 95.8% | 222 KB | 40 KB |
| ArguAna | 8,674 | 1,406 | 0.4995 | 0.4910 | 98.3% | 195 KB | 48 KB |
| FiQA | 57,638 | 648 | 0.3662 | 0.3309 | 90.3% | 678 KB | 64 KB |
| NQ | 2,681,468 | First 300 of 3,452 | 0.3911 | 0.3745 | 95.8% | 5.1 MB | 320 KB |

FiQA gains the most from retrieving more pages, reaching 94.9% with 8.

## What makes that possible

The upload grows with the number of clusters, so the question is how few clusters can be searched. This
table shows the share of exact nDCG@10 through the real protocol, with 4 pages each time:

| Clusters searched, and how they are chosen | SciFact | ArguAna | FiQA |
| --- | --- | --- | --- |
| 1, by a single centre per cluster | 59.9% | 76.9% | 58.6% |
| 1, by 32 prototypes per cluster | 67.0% | 87.9% | 62.3% |
| 2, by 32 prototypes per cluster | 83.7% | 96.0% | 75.8% |
| 4, by 32 prototypes per cluster | 91.3% | 97.8% | 86.9% |
| **8, by 32 prototypes per cluster (the default)** | **95.8%** | **98.3%** | **90.3%** |
| 16, by 32 prototypes per cluster | 96.7% | 98.2% | 93.3% |

Prototypes are downloaded once and used on your device, so they cost nothing per search. Each extra
cluster adds one query's worth of upload.

## Probes and pages together

| Probes | Pages | SciFact | ArguAna | FiQA | NQ |
| --- | --- | --- | --- | --- | --- |
| 4 | 4 | 91.3% | 97.8% | 86.9% | |
| 4 | 8 | 92.4% | 97.5% | 90.6% | |
| 8 | 2 | 92.9% | 92.9% | 83.2% | |
| 8 | 4 | 95.8% | 98.3% | 90.3% | 95.8% |
| 8 | 8 | 96.8% | 98.2% | 94.9% | 96.8% |
| 16 | 4 | 96.7% | 98.2% | 93.3% | 96.1% |
| 16 | 8 | 97.7% | 98.1% | 98.0% | 97.8% |

## Speed and traffic

`unsaid bench` against `unsaid serve` over loopback. Times are medians in milliseconds.

| | Demo sample | Simple English Wikipedia | BEIR NQ |
| --- | --- | --- | --- |
| Documents | 20,000 | 346,616 | 2,681,468 |
| Index build, with embeddings ready | 7 s | 11 s | 7 min |
| One-time download | 36.6 MB | 49.9 MB | 97.7 MB |
| Upload per search | 441 KB | 2.3 MB | 5.1 MB |
| Download per search | 48 KB | 96 KB | 320 KB |
| Server memory | 62 MB | 350 MB | 1.9 GB |
| **Server time, 10 cores** | **6** | **32** | **99** |
| **Server time, 1 core** | **11** | **82** | **405** |
| Embedding a short question | 5 | 5 | 5 |
| Encryption, unless prepared ahead | 77 | 311 | 743 |
| Decryption | 4 | 4 | 9 |
| Hold that hides timing | 246 | 246 | 246 |
| **Whole search** | **338** | **598** | **1,105** |

One core serves about 12 searches a second over the Wikipedia index, and about 2.5 a second over NQ. In
each search the server reads its whole index, so its time grows with the corpus.

`unsaid proxy` and `unsaid mcp` do their encryption in the background, so a search through them skips that
row. The hold is what keeps a query from revealing how long it took to process, and it makes up most of
the time on small indexes.

## In a browser

Chromium on the same machine, using the WebAssembly client with a single thread and the 20,000 article
demo index, with the server in a separate worker. Encryption is prepared ahead while the page is idle,
at 358 ms per search. After that, a search takes about 50 ms: embedding 4 to 18 ms, encryption 2 ms or
less, the server 13 to 23 ms, and decryption 2 to 6 ms.

## For context

[Tiptoe](https://eprint.iacr.org/2023/1438) reports searching 360 million pages with 145 core-seconds of
server compute, 56.9 MiB of communication and 2.7 seconds of latency. That is more than a hundred times
the corpus on a large cluster and on a different dataset, so these numbers are not directly comparable.

## Reproducing this

```sh
# Quality. Dataset links are in the BEIR repository.
unsaid eval path/to/scifact --out scifact-index --probe-counts 2,4,8,16 --page-counts 2,4,8
unsaid eval path/to/scifact --out scifact-base --prototypes 1 --probe-counts 1 --page-counts 4
unsaid eval path/to/nq --out nq-index --probe-counts 8,16 --page-counts 4,8 --max-queries 300

# Speed. Wikipedia comes from https://huggingface.co/datasets/wikimedia/wikipedia
uv run --with pyarrow scripts/prepare_wikipedia.py simple.parquet -o wiki.jsonl
uv run --with sentence-transformers scripts/embed_fast.py wiki.jsonl -o wiki.f32
unsaid build wiki.jsonl --out wiki-index --embeddings wiki.f32
unsaid serve wiki-index &
unsaid bench -n 50
RAYON_NUM_THREADS=1 unsaid serve wiki-index --port 8379 &
unsaid bench -n 30 --server http://127.0.0.1:8379

# The browser.
uv run --with playwright scripts/test_web.py http://127.0.0.1:8377
```

`scripts/embed_fast.py` needs a GPU to be quick. `unsaid build` and `unsaid eval` embed on the CPU if you
give them no embeddings, at around 50 documents a second on this machine. NQ used about 7.6 GB of memory
during evaluation.
