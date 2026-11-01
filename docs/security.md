# Security

**This has not had an independent audit.** The cryptographic design follows published work, and the
choices below are explained so that you can check them, but do not rely on it where a mistake could
hurt someone until experts have reviewed it. If you find a problem, see [SECURITY.md](../SECURITY.md).

## What is protected

The server cannot read what you searched for, which clusters your device chose, which documents or
pages you received, or whether two searches were the same. This rests on the learning with errors
problem being hard, and not on a policy, on deleted logs, or on trusting a third party.

Encryption is only part of it. A dishonest server can also watch what your client does, so the rest of
this page is about closing those channels, and about the ones that remain open.

## What is not protected

Read this part carefully. These things stay visible:

| Visible to the server | How to reduce it |
| --- | --- |
| Your IP address | Use a VPN or Tor |
| The time of each search | Nothing here hides this |
| How many searches you make, and how far apart | Send decoy searches at a steady rate |
| That you use this index at all | Choose a broad index. Someone using an index only about one illness reveals that topic even though each query is hidden |
| Which result you open, if you follow its link | Result links can point anywhere, including back at the server with a tag. Read the text on your device, and check where a link goes |

Unsaid also does not protect:

- **You from a false answer.** The server cannot read your query, but it can return wrong scores or
  serve a different index. Page hashes catch a page that differs from the published index, and `--pin`
  catches a swapped index, but nothing proves that the scores were computed honestly.
- **The server's documents from its users.** A client can eventually read everything by asking for
  each page. The hints also leak linear combinations of the data, and an index with fewer columns than
  the lattice dimension can be recovered from its hint outright. Only build an index from data the
  people searching it are allowed to read.
- **You from a malicious web page.** If you open the web interface from the server you are hiding from,
  that server supplies the code that handles your question, and it could send that in plain text. If
  you do not trust the operator, use the command line client or the gateway, which are programs on your
  own machine. The page's content security policy stops other sites receiving data, and result text is
  never inserted as HTML, but neither helps against the operator of the page.
- **You from your own device.** Malware on your machine can read your query. Lattice secrets are wiped
  from memory after use on a best-effort basis, since Rust can leave copies when values move, and the
  query itself is kept in ordinary memory.

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

`n = 1024` with these `q` and `sigma` is common in PIR papers, which estimate it with more detailed cost
models that count more of an attack's real expense. Queries may be stored for a long time and be
sensitive for a long time, so Unsaid uses the same conservative model as ML-KEM and clears ML-KEM-512's
margin in it. That costs about 25% more in hint size and client work. The estimate reflects attacks as
understood today and is not a proof. To go further, run the parameters through the
[lattice estimator](https://github.com/malb/lattice-estimator).

The attacker sees one sample per column for each query, far more than the attacks above can use, and
each query has a fresh secret. All queries against an index share one matrix `A`. An attacker could
reduce a basis for `A` once and reuse the work, which is why `A` depends on the index content.

## Defences against a malicious server

The design assumes that the server is an adversary and controls everything it sends.

| Attack | Defence |
| --- | --- |
| Weaken the encryption with chosen parameters | Parameters are compiled into the client, and a manifest that disagrees is refused |
| Choose a public matrix with a trapdoor | The client derives `A` by hashing the index contents |
| Send a model that crashes or stalls on certain words, so that a missing request reveals them | The command line client, gateway and MCP tool refuse any model not on a compiled-in list unless the user names its hash. The web page does too, unless the same server supplied the page. The parser uses checked arithmetic and validates every shape, and a test confirms that thousands of randomly damaged files never panic |
| Choose sizes that overflow in a browser's 32-bit arithmetic, to crash on chosen clusters | Every size is checked in 64 bits against strict limits before anything sized by it is allocated |
| Time the gap before the second request, which could depend on the chosen cluster's size or the results | Every row of every answer is decrypted whatever the cluster's size, every entry is visited when choosing pages, and the request is sealed and sent only after twice the fixed decryption time |
| Time how long a client takes to embed and route, which depends on the question | The command line client, gateway and MCP tool send each query on a multiple of 250 ms after the question arrived. The web page does not, because a person's typing gives the server no reference point |
| Time when the client closes its connection, which would show how long the results took to process | Connections close as soon as each answer has been read |
| Learn from how the client reacts to a bad answer | Every search has the same shape, including when pages fail their checks, and each request has a fresh secret, so there is no decryption oracle |
| Swap a page by adding to the answer, which needs no key | Pages are checked against a 128-bit digest, and a test performs exactly this attack |
| Make the client use one secret twice | A secret is consumed by encrypting, so reuse does not compile, and there is a test proving that it does not |
| Exhaust the client with a huge or endless response, or a page full of empty documents | Responses are cut off at the exact size expected, requests have overall time limits, and a page may not hold more documents than it has bytes |
| Redirect a query somewhere else | Redirects are refused |
| Use a cache path such as `/dev/zero` as a file's hash | A hash must be 64 hexadecimal characters |
| Attack the terminal with escape sequences in result text | Control characters and direction overrides are removed before printing |
| Have a web page use the local gateway | The gateway refuses requests from browsers and unexpected hosts |
| Serve different content, or a different traffic shape, to one person | Publish the `index_id`, which covers the whole manifest, and use `--pin` |

`unsafe` is forbidden in every crate except the WebAssembly bindings, where the generated glue needs it.

## Known weaknesses

- **Recognising a returning client.** A client downloads public files only once. A server could hand
  each client unique files and see, from what it fails to download later, that it has met this client
  before, even from a new IP address. If you are separating identities, use `--pin`, or a separate
  `UNSAID_CACHE` directory for each one.
- **Timing is reduced, not eliminated.** Ordinary jitter in your machine remains, and a question long
  enough to take more than 250 ms to process lands in a later window, which a slower machine reaches
  sooner. Nobody has measured what is left over a real network, and that would make a good project.
- **Your reactions.** If a page fails its check and you search again, the server sees a second search.
  It could use that to test a guess about which page you asked for, one guess at a time.
- **AI agents.** Results are text from the server, and a dishonest one could include instructions aimed
  at an agent, such as to send the query somewhere else. The tool's description says to treat results
  as data, but that is not a guarantee. Be careful about what else an agent using this can do.
- **Local side channels.** The Gaussian sampler reads its whole table for every sample, but the client
  as a whole makes no attempt to resist someone measuring timing or power on your own device.
- **Denial of service.** The server does expensive work for anyone who asks. It limits the queries per
  request, the memory held by uploads, how long uploads take and its concurrency, but not connections
  or the memory of answers waiting to be sent, so put rate limits in front of it.
- **Multi-target attacks.** Security estimates are per query. With very many queries against one `A`,
  a lattice attacker could benefit somewhat from precomputation.

## Implementation notes

- Randomness comes from the operating system through `getrandom`, and it keys a ChaCha20 generator.
- `A` is expanded with ChaCha12 from a public seed. It only needs to behave like a random oracle.
- The answer's noise bound is computed from the actual data when an index is built. It is stored
  in the manifest, and `unsaid info` shows it. A build fails if the bound is not met.
- `Matrix::apply`, the server's only operation, reads every entry for every query and never branches
  on a ciphertext value.
