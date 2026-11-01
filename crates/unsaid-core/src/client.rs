//! The client's half of a search, with no I/O in it.
//!
//! A search moves through two stages, each of which consumes the previous one:
//!
//! ```text
//! Index::begin(embedding, tokens)  -> (RankStage, request bytes)    send to /v1/rank
//! RankStage::receive(answer, ...)  -> (FetchStage, request bytes)   send to /v1/fetch
//! FetchStage::receive(answer)      -> Outcome
//! ```
//!
//! The traffic has a fixed shape. Every search sends the same number of requests of the same
//! sizes, whatever the query was and whatever came back, including when a page fails its
//! integrity check. A server that answers wrongly on purpose therefore cannot learn anything
//! from how the client reacts.

use sha2::{Digest, Sha256};
use unsaid_lwe::{LWE_DIM, Prepared, PublicMatrix, Secret};

use crate::manifest::{Manifest, ManifestError};
use crate::pages::{self, Document};

/// Bytes kept from the SHA-256 of a page.
pub const PAGE_DIGEST_BYTES: usize = 8;

/// What can go wrong during a search.
#[derive(Debug)]
pub enum SearchError {
    /// The index is malformed or does not match its manifest.
    Index(String),
    /// The server's answer has the wrong length.
    Answer(String),
    /// The caller supplied the wrong number or kind of prepared tokens.
    Tokens(String),
    /// The operating system could not supply randomness.
    Random(String),
}

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Index(m) => write!(f, "bad index: {m}"),
            Self::Answer(m) => write!(f, "bad answer from server: {m}"),
            Self::Tokens(m) => write!(f, "bad tokens: {m}"),
            Self::Random(m) => write!(f, "no randomness available: {m}"),
        }
    }
}

impl std::error::Error for SearchError {}

impl From<ManifestError> for SearchError {
    fn from(e: ManifestError) -> Self {
        Self::Index(e.0)
    }
}

fn check_blob(manifest: &Manifest, name: &str, data: &[u8]) -> Result<(), SearchError> {
    let info = manifest
        .blobs
        .get(name)
        .ok_or_else(|| SearchError::Index(format!("no blob named {name}")))?;
    if info.bytes != data.len() as u64 || info.sha256 != crate::manifest::sha256_hex(data) {
        return Err(SearchError::Index(format!("{name} does not match the manifest")));
    }
    Ok(())
}

fn words(data: &[u8]) -> Vec<u32> {
    data.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn floats(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Serialise words little-endian, the wire format of queries and answers.
pub fn to_wire(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The public files of an index.
pub struct IndexFiles<'a> {
    /// Projection and routing prototypes.
    pub routing: &'a [u8],
    /// Cluster sizes and page boundaries.
    pub layout: &'a [u8],
    /// Truncated SHA-256 of every page.
    pub page_digests: &'a [u8],
    /// Hint for the ranking matrix.
    pub rank_hint: &'a [u8],
    /// Hint for the document matrix.
    pub payload_hint: &'a [u8],
}

/// A verified index, ready to search.
pub struct Index {
    manifest: Manifest,
    mean: Vec<f32>,
    projection: Vec<f32>,
    prototypes: Vec<f32>,
    owners: Vec<u32>,
    cluster_start: Vec<u32>,
    page_first: Vec<u32>,
    page_digests: Vec<u8>,
    rank_hint: Vec<u32>,
    payload_hint: Vec<u32>,
    rank_a: PublicMatrix,
    payload_a: PublicMatrix,
}

/// How much traffic a search generates. All searches with one plan look identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchPlan {
    /// Clusters scored per search.
    pub probes: usize,
    /// Pages retrieved per search.
    pub pages: usize,
}

/// One search result.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Hit {
    /// Position of the document in the index.
    pub doc: u32,
    /// Estimated cosine similarity to the query.
    pub score: f32,
    /// The document itself.
    #[serde(flatten)]
    pub document: Document,
}

/// The result of a finished search.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Outcome {
    /// Best results first.
    pub hits: Vec<Hit>,
    /// Pages that failed their integrity check. Non-zero means the server misbehaved or a
    /// decryption failed, and those results were dropped.
    pub corrupted_pages: usize,
}

/// A search waiting for its ranking answer.
pub struct RankStage {
    clusters: Vec<usize>,
    secrets: Vec<Secret>,
}

/// A search waiting for its pages.
pub struct FetchStage {
    ranked: Vec<(u32, i32)>,
    pages: Vec<usize>,
    secrets: Vec<Secret>,
    limit: usize,
}

impl Index {
    /// Parse and verify an index. Every public file is checked against the manifest, and the
    /// manifest itself against the parameters this client is built with.
    pub fn open(manifest_json: &[u8], files: &IndexFiles<'_>) -> Result<Self, SearchError> {
        let manifest = Manifest::parse(manifest_json)?;
        check_blob(&manifest, "routing", files.routing)?;
        check_blob(&manifest, "layout", files.layout)?;
        check_blob(&manifest, "page_digests", files.page_digests)?;
        check_blob(&manifest, "rank_hint", files.rank_hint)?;
        check_blob(&manifest, "payload_hint", files.payload_hint)?;
        let bad = |m: &str| SearchError::Index(m.to_owned());

        let (d, md, t) = (manifest.dim, manifest.model_dim, manifest.prototypes);
        let expected_routing = (md + d * md + t * d) * 4 + t * 4;
        if files.routing.len() != expected_routing {
            return Err(bad("routing file has the wrong size"));
        }
        let (float_part, owner_part) = files.routing.split_at((md + d * md + t * d) * 4);
        let all = floats(float_part);
        let owners = words(owner_part);
        if owners.iter().any(|&o| o as usize >= manifest.clusters) {
            return Err(bad("a routing prototype names a cluster that does not exist"));
        }
        if all.iter().any(|v| !v.is_finite()) {
            return Err(bad("routing data contains a non-finite number"));
        }

        let cols = manifest.payload.cols;
        let c = manifest.clusters;
        if files.layout.len() != (c + cols) * 4 {
            return Err(bad("layout file has the wrong size"));
        }
        let layout = words(files.layout);
        let (sizes, page_first) = layout.split_at(c);
        let mut cluster_start = Vec::with_capacity(c + 1);
        let mut total = 0u64;
        cluster_start.push(0u32);
        for &size in sizes {
            if size as usize > manifest.rank.rows {
                return Err(bad("a cluster has more documents than the ranking matrix has rows"));
            }
            total += u64::from(size);
            if total > u64::from(u32::MAX) {
                return Err(bad("too many documents"));
            }
            cluster_start.push(total as u32);
        }
        if total != manifest.documents {
            return Err(bad("cluster sizes do not add up to the document count"));
        }
        if page_first.first() != Some(&0)
            || page_first.windows(2).any(|w| w[0] >= w[1])
            || page_first.last().is_some_and(|&l| u64::from(l) >= total.max(1))
        {
            return Err(bad("page boundaries are not increasing"));
        }
        if files.page_digests.len() != cols * PAGE_DIGEST_BYTES {
            return Err(bad("page digest file has the wrong size"));
        }
        let hint_len = |rows: usize| rows * LWE_DIM * 4;
        if files.rank_hint.len() != hint_len(manifest.rank.rows)
            || files.payload_hint.len() != hint_len(manifest.payload.rows)
        {
            return Err(bad("a hint has the wrong size"));
        }

        Ok(Self {
            rank_a: manifest.rank_matrix(),
            payload_a: manifest.payload_matrix(),
            mean: all[..md].to_vec(),
            projection: all[md..md + d * md].to_vec(),
            prototypes: all[md + d * md..].to_vec(),
            owners,
            cluster_start,
            page_first: page_first.to_vec(),
            page_digests: files.page_digests.to_vec(),
            rank_hint: words(files.rank_hint),
            payload_hint: words(files.payload_hint),
            manifest,
        })
    }

    /// The verified manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// The traffic shape the index recommends.
    pub fn default_plan(&self) -> SearchPlan {
        SearchPlan { probes: self.manifest.probes, pages: self.manifest.pages }
    }

    /// Prepare one token for a ranking query. This is the slow part of a search and can be
    /// done ahead of time. A token must never be saved to disk or used twice.
    pub fn prepare_rank(&self) -> Result<Prepared, SearchError> {
        Prepared::new(&self.rank_a).map_err(|e| SearchError::Random(e.to_string()))
    }

    /// Prepare one token for a page fetch. The same rules apply as for [`Index::prepare_rank`].
    pub fn prepare_fetch(&self) -> Result<Prepared, SearchError> {
        Prepared::new(&self.payload_a).map_err(|e| SearchError::Random(e.to_string()))
    }

    /// Project a model embedding into the index's space and normalise it.
    pub fn project(&self, embedding: &[f32]) -> Vec<f32> {
        let (d, md) = (self.manifest.dim, self.manifest.model_dim);
        assert_eq!(embedding.len(), md, "embedding has the wrong length");
        let mut out = vec![0f32; d];
        for (o, row) in out.iter_mut().zip(self.projection.chunks_exact(md)) {
            *o = row
                .iter()
                .zip(embedding.iter().zip(&self.mean))
                .map(|(p, (e, m))| p * (e - m))
                .sum();
        }
        let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        out.iter_mut().for_each(|v| *v /= norm);
        out
    }

    /// The `probes` clusters most likely to hold the best results, best first. A cluster's
    /// score is that of its best prototype. This uses only public data and runs on the client.
    pub fn route(&self, projected: &[f32], probes: usize) -> Vec<usize> {
        let d = self.manifest.dim;
        let mut best = vec![f32::NEG_INFINITY; self.manifest.clusters];
        for (proto, &owner) in self.prototypes.chunks_exact(d).zip(&self.owners) {
            let score: f32 = proto.iter().zip(projected).map(|(a, b)| a * b).sum();
            let slot = &mut best[owner as usize];
            if score > *slot {
                *slot = score;
            }
        }
        let mut order: Vec<usize> = (0..best.len()).collect();
        order.sort_by(|&a, &b| best[b].total_cmp(&best[a]).then(a.cmp(&b)));
        order.truncate(probes);
        order
    }

    /// Start a search. Uses one token per probe and returns the bytes to send to `/v1/rank`.
    pub fn begin(
        &self,
        embedding: &[f32],
        tokens: Vec<Prepared>,
    ) -> Result<(RankStage, Vec<u8>), SearchError> {
        let m = &self.manifest;
        if tokens.is_empty() || tokens.iter().any(|t| t.len() != m.rank.cols) {
            return Err(SearchError::Tokens("tokens do not fit the ranking matrix".into()));
        }
        let projected = self.project(embedding);
        let plain: Vec<i32> =
            projected.iter().map(|v| (v * m.query_scale).round() as i32).collect();
        // Route to as many clusters as there are tokens. If the index has fewer clusters than
        // that, the extra queries repeat one, so the traffic keeps its shape.
        let mut clusters = self.route(&projected, tokens.len());
        while clusters.len() < tokens.len() {
            clusters.push(clusters[0]);
        }
        let mut request = Vec::with_capacity(tokens.len() * m.rank.cols * 4);
        let mut secrets = Vec::with_capacity(tokens.len());
        for (token, &cluster) in tokens.into_iter().zip(&clusters) {
            let entries: Vec<(usize, i32)> =
                plain.iter().enumerate().map(|(i, &v)| (cluster * m.dim + i, v)).collect();
            let (secret, ct) = token.seal(m.rank.log_p, &entries);
            request.extend(ct.iter().flat_map(|v| v.to_le_bytes()));
            secrets.push(secret);
        }
        Ok((RankStage { clusters, secrets }, request))
    }

    fn page_of(&self, doc: u32) -> usize {
        self.page_first.partition_point(|&first| first <= doc) - 1
    }

    fn page_range(&self, page: usize) -> (u32, u32) {
        let end =
            self.page_first.get(page + 1).copied().unwrap_or(self.manifest.documents as u32);
        (self.page_first[page], end)
    }
}

impl RankStage {
    /// Clusters this search asked about. Known only to the client.
    pub fn clusters(&self) -> &[usize] {
        &self.clusters
    }

    /// Decrypt the scores, choose pages, and produce the bytes to send to `/v1/fetch`.
    /// Uses one token per page. `limit` caps how many results are returned in the end.
    pub fn receive(
        self,
        index: &Index,
        answer: &[u8],
        tokens: Vec<Prepared>,
        limit: usize,
    ) -> Result<(FetchStage, Vec<u8>), SearchError> {
        let m = index.manifest();
        if tokens.is_empty() || tokens.iter().any(|t| t.len() != m.payload.cols) {
            return Err(SearchError::Tokens("tokens do not fit the document matrix".into()));
        }
        let rows = m.rank.rows;
        if answer.len() != self.secrets.len() * rows * 4 {
            return Err(SearchError::Answer(format!(
                "expected {} bytes, got {}",
                self.secrets.len() * rows * 4,
                answer.len()
            )));
        }
        let answer = words(answer);
        let mut ranked: Vec<(u32, i32)> = Vec::new();
        let mut seen = Vec::with_capacity(self.clusters.len());
        for ((secret, &cluster), part) in
            self.secrets.iter().zip(&self.clusters).zip(answer.chunks_exact(rows))
        {
            // Padding rows of a smaller cluster carry nothing, so they need no decryption.
            let (start, end) = (index.cluster_start[cluster], index.cluster_start[cluster + 1]);
            let size = (end - start) as usize;
            if seen.contains(&cluster) {
                continue;
            }
            seen.push(cluster);
            let scores = secret.open_signed(
                m.rank.log_p,
                &part[..size],
                &index.rank_hint[..size * LWE_DIM],
            );
            ranked.extend(scores.iter().enumerate().map(|(row, &s)| (start + row as u32, s)));
        }
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        // Take pages in order of their best document, until the budget is used up.
        let budget = tokens.len();
        let mut pages: Vec<usize> = Vec::with_capacity(budget);
        for &(doc, _) in &ranked {
            if pages.len() == budget {
                break;
            }
            let page = index.page_of(doc);
            if !pages.contains(&page) {
                pages.push(page);
            }
        }
        // Too few results still produce a full set of fetches, so the shape stays fixed.
        let mut filler = 0;
        while pages.len() < budget {
            if !pages.contains(&filler) || filler + 1 >= m.payload.cols {
                pages.push(filler);
            }
            filler = (filler + 1).min(m.payload.cols - 1);
        }

        let mut request = Vec::with_capacity(budget * m.payload.cols * 4);
        let mut secrets = Vec::with_capacity(budget);
        for (token, &page) in tokens.into_iter().zip(&pages) {
            let (secret, ct) = token.seal(m.payload.log_p, &[(page, 1)]);
            request.extend(ct.iter().flat_map(|v| v.to_le_bytes()));
            secrets.push(secret);
        }
        Ok((FetchStage { ranked, pages, secrets, limit }, request))
    }
}

impl FetchStage {
    /// Pages this search asked for. Known only to the client.
    pub fn pages(&self) -> &[usize] {
        &self.pages
    }

    /// Decrypt the pages, check them and assemble the results.
    pub fn receive(self, index: &Index, answer: &[u8]) -> Result<Outcome, SearchError> {
        let m = index.manifest();
        let rows = m.payload.rows;
        if answer.len() != self.secrets.len() * rows * 4 {
            return Err(SearchError::Answer(format!(
                "expected {} bytes, got {}",
                self.secrets.len() * rows * 4,
                answer.len()
            )));
        }
        let answer = words(answer);
        let mut documents: Vec<(u32, Document)> = Vec::new();
        let mut corrupted_pages = 0;
        let mut done: Vec<usize> = Vec::new();
        for ((secret, &page), part) in
            self.secrets.iter().zip(&self.pages).zip(answer.chunks_exact(rows))
        {
            if done.contains(&page) {
                continue;
            }
            done.push(page);
            let bytes = secret.open_bytes(part, &index.payload_hint);
            let digest = Sha256::digest(&bytes);
            let at = page * PAGE_DIGEST_BYTES;
            if digest[..PAGE_DIGEST_BYTES] != index.page_digests[at..at + PAGE_DIGEST_BYTES] {
                corrupted_pages += 1;
                continue;
            }
            let (first, end) = index.page_range(page);
            match pages::decode(&bytes) {
                Some(docs) if docs.len() == (end - first) as usize => {
                    documents.extend(docs.into_iter().enumerate().map(|(i, d)| (first + i as u32, d)));
                }
                _ => corrupted_pages += 1,
            }
        }
        let scale = m.doc_scale * m.query_scale;
        let hits = self
            .ranked
            .iter()
            .filter_map(|&(doc, score)| {
                let found = documents.iter().position(|(id, _)| *id == doc)?;
                Some(Hit {
                    doc,
                    score: score as f32 / scale,
                    document: documents.swap_remove(found).1,
                })
            })
            .take(self.limit)
            .collect();
        Ok(Outcome { hits, corrupted_pages })
    }
}
