# The Unsaid protocol, version 1

This is everything needed to write a compatible client or server.
[`tutorial/3_real_client.py`](../tutorial/3_real_client.py) is a second implementation, written from
this document, that shares no code with the Rust client.

Integers are little-endian. Floats are IEEE 754 single precision. Arithmetic on ciphertexts, hints
and answers is modulo 2^32. `||` means concatenation.

## 1. Lattice parameters

| Parameter | Value |
| --- | --- |
| Secret dimension `n` | 1280 |
| Modulus `q` | 2^32 |
| Error | Discrete Gaussian, standard deviation 6.4 |

These are fixed by the protocol. A client must refuse a manifest that states anything else, because a
server that could choose weaker parameters could read queries. See [security.md](security.md).

## 2. The manifest

`GET /v1/manifest` returns a JSON object:

| Field | Meaning |
| --- | --- |
| `format` | Always `unsaid-index-1` |
| `name` | Display name |
| `documents` | Number of documents |
| `model` | Embedding model name |
| `model_dim`, `dim` | Vector length before and after projection |
| `clusters` | Number of clusters |
| `prototypes` | Number of routing prototypes, across all clusters |
| `doc_scale`, `query_scale` | Multipliers applied to vectors before rounding |
| `probes`, `pages` | The recommended number of encrypted queries per search |
| `rank` | The ranking matrix: `rows`, `cols`, `log_p`, `failure_log2` |
| `payload` | The document matrix, with the same fields. `log_p` is always 8 |
| `lwe` | `{"n": 1280, "log_q": 32, "sigma": 6.4}` |
| `blobs` | `model`, `routing`, `layout`, `page_digests`, `rank_hint`, `payload_hint`, each with `sha256` and `bytes` |

`rank.cols` must equal `clusters * dim`.

### Identifiers

```text
numbers    = the following as text, joined by single spaces:
             format documents model model_dim dim clusters prototypes doc_scale query_scale
             rank.rows rank.cols rank.log_p payload.rows payload.cols payload.log_p
             (floats in their shortest form, so 64 and not 64.0)

content_id = SHA-256( "unsaid/v1/content\0" || numbers
                      || for name in model, page_digests, layout, routing:
                         "\0" || name || "=" || hex(blobs[name].sha256) )

index_id   = hex of SHA-256( "unsaid/v1/index\0" || the manifest file, byte for byte )
```

`index_id` covers everything in the manifest and, through the hashes in it, every public file.
Publishing it elsewhere lets clients detect a swapped index. See `--pin` in
[deploying.md](deploying.md).

### What a client must check

The server supplies the manifest, so it can put anything in there. A client must refuse a manifest
unless all of these hold, doing the arithmetic in at least 64 bits:

| Check | Limit |
| --- | --- |
| Lattice parameters | Exactly as in section 1 |
| `rank.rows`, `payload.rows` | 1 to 65,536 |
| `rank.cols`, `payload.cols` | 1 to 16,777,216 |
| `rank.cols` | Equal to `clusters * dim` |
| `rank.log_p` | 2 to 24 |
| `payload.log_p` | 8 |
| `probes` | 1 to the smaller of `clusters` and 64 |
| `pages` | 1 to 64 |
| `probes * rank.cols + pages * payload.cols` | At most 67,108,864 |
| `dim` and `model_dim` | `dim` at least 1, at most `model_dim`, at most 8,192 |
| `prototypes` | 1 to 1,048,576 |
| `clusters` | 1 to 16,777,216 |
| `name`, `model` | At most 1,024 bytes |
| The manifest file | At most 1 MiB |
| `documents` | At least `payload.cols`, below 2^32 - 1 |
| `doc_scale`, `query_scale` | 0.001 to 1,000,000 |
| Each `blobs[name].sha256` | Exactly 64 lowercase hexadecimal characters |
| Each `blobs[name].bytes` | The size in section 3 |
| The model file | At most 2 GiB |
| All public files together | At most 8 GiB |

**A client should not use a model it does not already trust.** A model shapes how a query is processed
on the client, so one built to crash or slow down on certain words could reveal them, with no attack on
the encryption at all. The command line client, the gateway and the MCP tool refuse any model whose hash
is not compiled in, unless the user names that hash. The web page does the same, unless the same server
supplied the page, because that page already had to be trusted just as much.

### Public matrices

The server must not get to choose `A`, because a chosen `A` could have a trapdoor. So the client
derives it from the index:

```text
seed(label) = SHA-256( "unsaid/v1/matrix-a\0" || label || "\0" || content_id )
```

The label is `rank` or `payload`. `A` has `cols` rows and 1280 columns. Row `j` is the first 1280
words of the ChaCha keystream (the RFC 8439 construction) with 12 rounds, key `seed(label)`, a 32-bit
block counter starting at 0, and nonce words `(j mod 2^32, j >> 32, 0)`.

For checking an implementation, with an all-zero seed and row 0, the first four words are
`0x6a9af49b 0x53f95507 0x12ce1f81 0xd583265f`.

## 3. Public files

`GET /v1/blob/{sha256}` returns a file by its hash. Clients must check the hash, and may cache files
forever.

- **`model`** is the embedding model in the `USM2` format, defined in
  [`crates/unsaid-embed/src/model.rs`](../crates/unsaid-embed/src/model.rs).
- **`routing`** holds `mean` (`model_dim` floats), `projection` (`dim * model_dim` floats, by row),
  `prototypes` (`prototypes * dim` floats, by row), and `owners` (`prototypes` 32-bit cluster numbers).
- **`layout`** holds `sizes` (`clusters` 32-bit document counts) and `page_first` (`payload.cols`
  32-bit document numbers, strictly increasing and starting at 0). Documents are numbered cluster by
  cluster.
- **`page_digests`** holds the first 16 bytes of the SHA-256 of each page.
- **`rank_hint`** and **`payload_hint`** each hold the `rows * 1280` words of `H = M A`, by row.

## 4. Encryption

To encrypt a plaintext `x` in `Z_p`, with `p = 2^log_p`, for a matrix with `cols` columns:

```text
s  = 1280 uniform random words, from a cryptographic generator, never used twice
e  = cols samples of the error distribution
b  = A s + e + 2^(32 - log_p) * x
```

To decrypt an answer `a`, using hint `H`:

```text
x' = (a - H s + 2^(31 - log_p)) >> (32 - log_p)
```

Matrix entries are signed bytes in the hint and in the server's product. Ranking results are read as
signed values in `-p/2 .. p/2`. Document results are read as bytes.

Reusing `s` reveals the difference of two plaintexts, so a secret must only ever be used once.

## 5. A search

1. **Embed.** Compute the query's vector `v` with the model. The client does this itself.
2. **Project.** `u = projection * (v - mean)`, then scale `u` to length 1.
3. **Route.** A prototype's score is its dot product with `u`, and a cluster's score is that of its
   best prototype. Take the `probes` best clusters, breaking ties by the lower number. If the index
   has fewer clusters than probes, repeat the first so that the traffic keeps its shape.
4. **Rank.** For each chosen cluster `c`, encrypt the vector that has `round(u[i] * query_scale)` at
   column `c * dim + i` and zero elsewhere, with `log_p = rank.log_p`. `POST /v1/rank` the
   concatenated ciphertexts.
5. **Read the scores.** The answer holds `probes * rank.rows` words. In each block, row `r` is the
   score of document `start(c) + r`. Rows past the cluster's size are padding, and a client must
   decrypt them anyway, because the server chooses cluster sizes and can time what comes next. A
   document's estimated similarity is its value divided by `doc_scale * query_scale`.
6. **Choose pages.** Walk the documents from best to worst and collect each one's page, until there
   are `pages` distinct pages. If there are too few, add others so there is always the same number.
   **Wait** until twice the time step 5 took before going on, because the time this step takes
   depends on the results and the server can measure the gap.
7. **Fetch.** For each page `p`, encrypt the unit vector with 1 at column `p` and `log_p = 8`.
   `POST /v1/fetch` the concatenated ciphertexts.
8. **Open.** The answer holds `pages * payload.rows` words. Decrypt each block to `payload.rows` bytes,
   compare its truncated SHA-256 with `page_digests`, and decode it, refusing any page that holds more
   documents than the layout says. If a check fails, drop that page, but do not change anything a
   server could observe.

Steps 1 to 3 depend on the question too, and a server has a reference point whenever a program has just
connected or is searching in a loop. So the reference client does all the work that does not depend on
the question first, then runs steps 1 to 4 and sends the query on a multiple of 250 ms after the
question arrived. The web page skips this, because a person's typing gives the server no reference point.

A client should also close its connection as soon as it has read each answer. A connection left open
closes later, after the results have been processed, and the server could time that.

## 6. Pages

A page is exactly `payload.rows` bytes:

```text
length (2 bytes) || raw DEFLATE stream of `length` bytes || zero padding
```

The inflated stream is a series of documents. Each is three strings, `title`, `text` and `url`, and
each string is its length as a little-endian base-128 varint followed by UTF-8 bytes. Page `p` holds
documents `page_first[p]`, `page_first[p] + 1`, and so on.

## 7. HTTP interface

| Request | Response |
| --- | --- |
| `GET /v1/manifest` | The manifest |
| `GET /v1/blob/{sha256}` | A public file, or 404 |
| `POST /v1/rank` | `application/octet-stream`: whole ciphertexts of `rank.cols` words, at most 64 |
| `POST /v1/fetch` | `application/octet-stream`: whole ciphertexts of `payload.cols` words, at most 64 |
| `GET /v1/health` | `ok` |

Queries must state a `Content-Length`, or they get a 411. A body that is empty, not a whole number of
ciphertexts or not the stated length gets a 400, and one with too many ciphertexts gets a 413. An upload
that takes longer than 120 seconds, or stalls for 10, gets a 408. A server out of memory for uploads
answers 503 rather than making anyone wait.

A client must not follow redirects, because one could send a query anywhere, and should not read more
than the exact size of the response it expects.
