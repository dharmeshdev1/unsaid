# Ideas to build on this

Each of these is a real open problem or a useful addition. They range from a weekend to a thesis,
so they suit course projects well, and each comes with a way to tell whether it worked. If you take
one on, open an issue so that others can find your work.

## Good first projects

| Idea | What you would do | How to measure it |
| --- | --- | --- |
| **Private RAG** | Connect `unsaid proxy` to a local language model, so that neither the question nor the retrieval leaves your machine in the clear. See [`examples/private_rag.py`](../examples/private_rag.py) | Answer quality against the same pipeline with plain retrieval |
| **Evaluate more datasets** | Run `unsaid eval` on other BEIR sets and add them to the benchmarks | Share of exact nDCG@10 |
| **Other languages** | Pack a multilingual model with `crates/unsaid-embed/examples/pack.rs` and build an index in your language | The same evaluation on a dataset in that language |
| **A client in another language** | Write Go, Swift or Kotlin from [the protocol](protocol.md) | It interoperates with the Rust server, and the test vectors pass |
| **Private search for your field** | Build an index over medical guidelines, case law or course notes, and write up who benefits | Whether people in that field find it useful |

## Security projects

| Idea | What you would do | How to measure it |
| --- | --- | --- |
| **Attack it** | Run a logging server and try to infer queries from sizes, timing and order | How far your guess beats chance, with real numbers |
| **Measure the timing gap** | Quantify what the delay between the two requests reveals, and fix it if it reveals anything | Mutual information between the delay and the chosen cluster |
| **Run the lattice estimator** | Run the parameters through the [lattice estimator](https://github.com/malb/lattice-estimator) and compare with `tools/lwe_estimate.py` | Bit security under each cost model |
| **Verifiable answers** | Let the client check that scores came from the published index | Proof size and client time |
| **Fuzz the parsers** | Fuzz the manifest, model and page parsers | Crashes found, and coverage reached |

## Harder research

| Idea | What you would do | How to measure it |
| --- | --- | --- |
| **Shrink the upload** | Replace raw LWE ciphertexts with a packed ring-based form, as in [YPIR](https://eprint.iacr.org/2024/270) | Bytes per query at the same security and quality |
| **Shrink the hint** | Compress the hint as DoublePIR does, or remove it | One-time download against server cost |
| **Better routing** | Learn the cluster assignment and prototypes so that fewer probes reach the same quality | nDCG@10 at 1, 2 and 4 probes |
| **Live updates** | Add documents without rebuilding, for example with a small second index merged at query time | Update cost and quality over time |
| **Faster servers** | Run the matrix product on a GPU or with matrix compression | Queries per second per dollar |
| **Private autocomplete** | Suggest completions as someone types, with a suitable index and traffic shape | Latency at each keystroke, and what the traffic reveals |
