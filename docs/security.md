# Security

**This has not had an independent audit.** The cryptographic design follows published work, and the
choices below are explained so that you can check them, but do not rely on it where a mistake could
hurt someone until experts have reviewed it. If you find a problem, see [SECURITY.md](../SECURITY.md).

## What is protected

The server, or anyone who takes it over, cannot learn:

- what you searched for,
- which clusters your device chose,
- which documents or pages you received, or
- whether two searches were the same.

That holds even if the server misbehaves on purpose, assuming that the learning with errors problem
is hard. It does not rest on a policy, on deleting logs, or on trusting a third party.

## What is not protected

Read this part carefully. These things stay visible:

| Visible to the server | How to reduce it |
| --- | --- |
| Your IP address | Use a VPN or Tor |
| The time of each search | Nothing here hides this |
| How many searches you make, and how far apart | Send decoy searches at a steady rate |
| That you use this index at all | Choose a broad index. Someone using an index only about one illness reveals that topic even though each query is hidden |
| Which result you open, if you follow its link | The destination site sees your visit. The text in the results is already on your device |

Unsaid also does not protect:

- **You from a false answer.** The server cannot read your query, but it can return wrong scores or
  serve a different index. Page hashes catch corruption relative to the published index, and
  `--pin` catches a swapped index, but nothing proves that the scores were computed honestly.
- **The server's documents from its users.** A client can eventually read everything by asking for
  each page. The hints also leak linear combinations of the data, and an index with fewer columns than
  the lattice dimension can be recovered from its hint outright. Only build an index from data the
  people searching it are allowed to read.
- **You from a malicious web page.** If you open the web interface from the server you are hiding from,
  that server supplies the code that handles your question. A dishonest operator could serve a page
  that sends it in plain text. Use the command line client, or host a copy of the page separately.
  The page's content security policy stops other sites receiving data, and result text is never
  inserted as HTML, but neither helps against the operator of the page.
- **Your device.** Malware on your own machine can read your query. Queries are held in ordinary
  memory, and only the lattice secrets are wiped after use.

## Lattice parameters

| Parameter | Value |
| --- | --- |
| Dimension `n` | 1280 |
| Modulus `q` | 2^32 |
| Error | Discrete Gaussian with standard deviation 6.4, cut off at 62 |

[`tools/lwe_estimate.py`](../tools/lwe_estimate.py) estimates hardness under the Core-SVP cost model,
which is the one the ML-KEM specification tabulates. It only counts one call to an SVP oracle, so
it is a pessimistic lower bound that is good for comparison rather than a statement of the real cost
of an attack. The script first reproduces the published figures and then applies the same formulas:

| Parameters | Primal, classical | Dual, classical | Primal, quantum |
| --- | --- | --- | --- |
| ML-KEM-512 (published 2^118) | 2^118.6 | 2^118.0 | 2^107.6 |
| ML-KEM-768 (published 2^182) | 2^182.2 | 2^181.0 | 2^165.4 |
| ML-KEM-1024 (published 2^256) | 2^255.2 | 2^253.5 | 2^231.6 |
| `n = 1024`, `q = 2^32`, `sigma = 6.4` | 2^95.2 | 2^95.2 | 2^86.4 |
| **Unsaid: `n = 1280`, `q = 2^32`, `sigma = 6.4`** | **2^128.8** | **2^128.5** | **2^116.9** |

`n = 1024` is common in PIR papers, and is reported at about 128 bits under the more detailed cost
model of the [lattice estimator](https://github.com/malb/lattice-estimator). Queries may be stored
for a long time and be sensitive for a long time, so Unsaid uses the same conservative model as
ML-KEM and clears ML-KEM-512's margin in it. That costs about 25% more in hint size and client work.
The estimate takes attacks as understood today and is not a proof.

The attacker sees one sample per column for each query, far more than the attacks above can use, and
each query has a fresh secret. All queries against an index share one matrix `A`. An attacker could
reduce a basis for `A` once and reuse the work, which is why `A` depends on the index content.

## Defences against a malicious server

The design assumes that the server is an adversary and controls everything it sends.

| Attack | Defence |
| --- | --- |
| Weaken the encryption with chosen parameters | Parameters are compiled into the client, and a manifest that disagrees is refused |
| Choose a public matrix with a trapdoor | The client derives `A` by hashing the index contents |
| Learn from how the client reacts to a bad answer | Every search has the same shape, including when pages fail their checks, and each request has a fresh secret, so there is no decryption oracle |
| Learn from which pages get requested | They are encrypted like everything else |
| Make the client use one secret twice | A secret is consumed by encrypting, so reuse does not compile, and there is a test proving that it does not |
| Exhaust the client's memory with a huge index or page | Sizes are bounded and checked, and page decompression is capped |
| Crash the client with malformed files | Parsers return errors and do not panic, including on the model file. `unsafe` is forbidden in every crate except the WebAssembly bindings, where the generated glue needs it |
| Serve different content to one person | Publish the `index_id` and use `--pin` |

## Known weaknesses

- **Timing between requests.** The second request leaves right after the first answer is processed.
  The client decrypts every row so that the work does not depend on the cluster, but sorting and
  page selection are not constant-time. The effect should be small next to network jitter, and
  nobody has measured it carefully. That would make a good project.
- **Local side channels.** The Gaussian sampler and the rest of the client make no attempt to resist
  an attacker who can measure timing or power on your own device.
- **Denial of service.** The server does expensive work for anyone who asks. Put rate limits in front
  of it. The server limits the queries per request and its concurrency, and nothing else.
- **Multi-target attacks.** Security estimates are per query. With very many queries against one `A`,
  a lattice attacker could benefit somewhat from precomputation.

## Implementation notes

- Randomness comes from the operating system through `getrandom`, and it keys a ChaCha20 generator.
- `A` is expanded with ChaCha12 from a public seed. It only needs to behave like a random oracle.
- The answer's noise bound is computed from the actual data when an index is built. It is stored
  in the manifest, and `unsaid info` shows it. A build fails if the bound is not met.
- `Matrix::apply`, the server's only operation, reads every entry for every query and never branches
  on a ciphertext value.
