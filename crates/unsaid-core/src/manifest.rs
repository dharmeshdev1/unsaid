//! The index manifest, and the checks a client makes before trusting it.
//!
//! The server supplies the manifest, so a malicious server controls everything in it. Two
//! things must therefore never be taken from it. The lattice parameters are compiled into
//! the client and any manifest that disagrees is rejected. And the seed of the public matrix
//! `A` is derived by the client from a hash, because a server free to choose `A` could pick
//! one with a trapdoor and decrypt every query.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use unsaid_lwe::{LOG_Q, LWE_DIM, PublicMatrix, SIGMA};

/// Format identifier written to and required from every manifest.
pub const FORMAT: &str = "unsaid-index-1";

/// Bytes kept from the SHA-256 of a page. Adding to an answer needs no key, so a server could
/// swap a page for any other with the same digest, which makes this a second preimage problem.
pub const PAGE_DIGEST_BYTES: usize = 16;

/// Longest ciphertext a client will agree to create, in entries.
pub const MAX_COLS: usize = 1 << 24;
/// Largest answer a client will agree to process, in entries.
pub const MAX_ROWS: usize = 1 << 16;
/// Most ciphertext entries in one search, which bounds the work a server can make a client do.
pub const MAX_QUERY_WORDS: u64 = 1 << 26;
/// Longest model vector a client will accept.
pub const MAX_MODEL_DIM: usize = 1 << 13;
/// Most routing prototypes a client will accept.
pub const MAX_PROTOTYPES: usize = 1 << 24;
/// Largest model file a client will accept.
pub const MAX_MODEL_BYTES: u64 = 1 << 31;
/// Most a client will download, once, for one index.
pub const MAX_PUBLIC_BYTES: u64 = 1 << 33;
/// Longest name a client will accept.
const MAX_NAME_BYTES: usize = 1 << 10;
/// Quantisation scales outside this range have no use and only make results meaningless.
const SCALE_RANGE: std::ops::RangeInclusive<f32> = 1e-3..=1e6;

/// Lattice parameters as recorded in a manifest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LweParams {
    /// Secret dimension.
    pub n: usize,
    /// Log of the ciphertext modulus.
    pub log_q: u32,
    /// Standard deviation of the error.
    pub sigma: f64,
}

impl LweParams {
    /// The only parameters this version of the protocol accepts.
    pub fn current() -> Self {
        Self {
            n: LWE_DIM,
            log_q: LOG_Q,
            sigma: SIGMA,
        }
    }
}

/// Shape of one private matrix.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MatrixInfo {
    /// Rows, which is the length of an answer.
    pub rows: usize,
    /// Columns, which is the length of a query.
    pub cols: usize,
    /// Log of the plaintext modulus.
    pub log_p: u32,
    /// Base-2 log of the bound on a per-answer decryption failure.
    pub failure_log2: f64,
}

/// A public file, identified by its hash.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlobInfo {
    /// Hex SHA-256 of the contents.
    pub sha256: String,
    /// Size in bytes.
    pub bytes: u64,
}

/// Everything a client needs to know about an index.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// Must equal [`FORMAT`].
    pub format: String,
    /// Display name.
    pub name: String,
    /// Number of documents.
    pub documents: u64,
    /// Embedding model identifier.
    pub model: String,
    /// Length of the model's vectors.
    pub model_dim: usize,
    /// Length of vectors after projection.
    pub dim: usize,
    /// Number of clusters.
    pub clusters: usize,
    /// Number of routing prototypes.
    pub prototypes: usize,
    /// Multiplier applied to document vectors before rounding.
    pub doc_scale: f32,
    /// Multiplier applied to query vectors before rounding.
    pub query_scale: f32,
    /// Encrypted queries per search.
    pub probes: usize,
    /// Pages retrieved per search.
    pub pages: usize,
    /// The ranking matrix.
    pub rank: MatrixInfo,
    /// The document matrix.
    pub payload: MatrixInfo,
    /// Lattice parameters.
    pub lwe: LweParams,
    /// Public files by name.
    pub blobs: BTreeMap<String, BlobInfo>,
}

/// Why a manifest was rejected.
#[derive(Debug, PartialEq)]
pub struct ManifestError(pub String);

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "manifest rejected: {}", self.0)
    }
}

impl std::error::Error for ManifestError {}

/// Hex encoding of a SHA-256 digest of `data`.
pub fn sha256_hex(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The identifier of an index, covering everything in its manifest and so, through the hashes in
/// there, every public file. Publish it out of band and clients can detect a swapped index.
pub fn index_id(manifest_json: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(b"unsaid/v1/index\0");
    h.update(manifest_json);
    hex(&h.finalize())
}

fn is_hash(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Names of the public files that define an index, as opposed to files derived from it.
const CONTENT_BLOBS: [&str; 4] = ["model", "page_digests", "layout", "routing"];
const HINT_BLOBS: [&str; 2] = ["payload_hint", "rank_hint"];

impl Manifest {
    /// Parse and validate a manifest.
    pub fn parse(json: &[u8]) -> Result<Self, ManifestError> {
        let manifest: Self =
            serde_json::from_slice(json).map_err(|e| ManifestError(format!("not valid: {e}")))?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Check everything a client can check before downloading anything else.
    pub fn validate(&self) -> Result<(), ManifestError> {
        let fail = |message: String| Err(ManifestError(message));
        if self.format != FORMAT {
            return fail(format!("unsupported format {:?}", self.format));
        }
        if self.lwe != LweParams::current() {
            return fail(format!(
                "lattice parameters {:?} are not the ones this client is built with",
                self.lwe
            ));
        }
        if self.name.len() > MAX_NAME_BYTES || self.model.len() > MAX_NAME_BYTES {
            return fail("a name is too long".into());
        }
        if self.dim == 0
            || self.model_dim == 0
            || self.dim > self.model_dim
            || self.model_dim > MAX_MODEL_DIM
        {
            return fail(format!(
                "bad dimensions {} and {}",
                self.dim, self.model_dim
            ));
        }
        if self.documents == 0 || self.documents >= u64::from(u32::MAX) {
            return fail(format!("bad document count {}", self.documents));
        }
        if self.clusters == 0
            || self.clusters > MAX_COLS
            || self.prototypes == 0
            || self.prototypes > MAX_PROTOTYPES
        {
            return fail("an index needs clusters and a sane number of prototypes".into());
        }
        if self.probes == 0 || self.probes > self.clusters.min(64) {
            return fail(format!("bad probe count {}", self.probes));
        }
        if self.pages == 0 || self.pages > 64 {
            return fail(format!("bad page count {}", self.pages));
        }
        if !SCALE_RANGE.contains(&self.doc_scale) || !SCALE_RANGE.contains(&self.query_scale) {
            return fail("quantisation scales are out of range".into());
        }
        for (label, m) in [("ranking", &self.rank), ("document", &self.payload)] {
            if m.rows == 0 || m.rows > MAX_ROWS || m.cols == 0 || m.cols > MAX_COLS {
                return fail(format!("{label} matrix has an unacceptable shape"));
            }
            if !(2..=24).contains(&m.log_p) {
                return fail(format!(
                    "{label} matrix has an unacceptable plaintext modulus"
                ));
            }
        }
        // All of this uses 64 bits, because a browser client has a 32-bit usize.
        if self.rank.cols as u64 != self.clusters as u64 * self.dim as u64 {
            return fail("ranking matrix does not have clusters * dim columns".into());
        }
        if self.payload.log_p != 8 {
            return fail("the document matrix must hold bytes".into());
        }
        if self.payload.cols as u64 > self.documents {
            return fail("there are more pages than documents".into());
        }
        let words = self.probes as u64 * self.rank.cols as u64
            + self.pages as u64 * self.payload.cols as u64;
        if words > MAX_QUERY_WORDS {
            return fail("one search would need too much ciphertext".into());
        }
        let mut total = 0u64;
        for name in CONTENT_BLOBS.iter().chain(&HINT_BLOBS) {
            let Some(blob) = self.blobs.get(*name) else {
                return fail(format!("the blob {name} is missing"));
            };
            if !is_hash(&blob.sha256) {
                return fail(format!("the blob {name} does not have a valid hash"));
            }
            let expected = self.expected_bytes(name);
            if expected.is_some_and(|bytes| bytes != blob.bytes) {
                return fail(format!("the blob {name} has the wrong size"));
            }
            total = total.saturating_add(blob.bytes);
        }
        if self.blobs["model"].bytes > MAX_MODEL_BYTES || total > MAX_PUBLIC_BYTES {
            return fail("the public files are too large".into());
        }
        Ok(())
    }

    /// The size a public file must have, given the rest of the manifest. The model's size is
    /// not implied by anything else, so that one has no answer.
    fn expected_bytes(&self, name: &str) -> Option<u64> {
        let (md, d, t) = (
            self.model_dim as u64,
            self.dim as u64,
            self.prototypes as u64,
        );
        let hint = |rows: usize| rows as u64 * LWE_DIM as u64 * 4;
        match name {
            "routing" => Some((md + d * md + t * d) * 4 + t * 4),
            "layout" => Some((self.clusters as u64 + self.payload.cols as u64) * 4),
            "page_digests" => Some(self.payload.cols as u64 * PAGE_DIGEST_BYTES as u64),
            "rank_hint" => Some(hint(self.rank.rows)),
            "payload_hint" => Some(hint(self.payload.rows)),
            _ => None,
        }
    }

    /// A hash of everything that defines this index except the hints, which are computed
    /// from it. Both public matrices are derived from this value.
    pub fn content_id(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"unsaid/v1/content\0");
        let numbers = format!(
            "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
            self.format,
            self.documents,
            self.model,
            self.model_dim,
            self.dim,
            self.clusters,
            self.prototypes,
            self.doc_scale,
            self.query_scale,
            self.rank.rows,
            self.rank.cols,
            self.rank.log_p,
            self.payload.rows,
            self.payload.cols,
            self.payload.log_p,
        );
        h.update(numbers.as_bytes());
        for name in CONTENT_BLOBS {
            h.update(b"\0");
            h.update(name.as_bytes());
            h.update(b"=");
            if let Some(blob) = self.blobs.get(name) {
                h.update(blob.sha256.as_bytes());
            }
        }
        h.finalize().into()
    }

    fn seed(&self, label: &str) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"unsaid/v1/matrix-a\0");
        h.update(label.as_bytes());
        h.update(b"\0");
        h.update(self.content_id());
        h.finalize().into()
    }

    /// The public matrix for the ranking database, derived by hash.
    pub fn rank_matrix(&self) -> PublicMatrix {
        PublicMatrix::new(self.seed("rank"), self.rank.cols)
    }

    /// The public matrix for the document database, derived by hash.
    pub fn payload_matrix(&self) -> PublicMatrix {
        PublicMatrix::new(self.seed("payload"), self.payload.cols)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn sample() -> Manifest {
        let blob = |s: &str| BlobInfo {
            sha256: sha256_hex(s.as_bytes()),
            bytes: 1,
        };
        let mut blobs = BTreeMap::new();
        for name in CONTENT_BLOBS.iter().chain(&HINT_BLOBS) {
            blobs.insert((*name).to_owned(), blob(name));
        }
        let mut manifest = Manifest {
            format: FORMAT.into(),
            name: "test".into(),
            documents: 10,
            model: "m".into(),
            model_dim: 8,
            dim: 4,
            clusters: 2,
            prototypes: 2,
            doc_scale: 64.0,
            query_scale: 64.0,
            probes: 1,
            pages: 1,
            rank: MatrixInfo {
                rows: 6,
                cols: 8,
                log_p: 14,
                failure_log2: -100.0,
            },
            payload: MatrixInfo {
                rows: 64,
                cols: 4,
                log_p: 8,
                failure_log2: -100.0,
            },
            lwe: LweParams::current(),
            blobs,
        };
        for name in CONTENT_BLOBS.iter().chain(&HINT_BLOBS) {
            if let Some(bytes) = manifest.expected_bytes(name) {
                manifest.blobs.get_mut(*name).unwrap().bytes = bytes;
            }
        }
        manifest
    }

    #[test]
    fn a_good_manifest_round_trips() {
        let m = sample();
        let json = serde_json::to_vec(&m).unwrap();
        assert_eq!(Manifest::parse(&json).unwrap(), m);
    }

    #[test]
    fn weakened_lattice_parameters_are_rejected() {
        for lwe in [
            LweParams {
                n: 16,
                ..LweParams::current()
            },
            LweParams {
                n: 1024,
                ..LweParams::current()
            },
            LweParams {
                sigma: 0.0,
                ..LweParams::current()
            },
            LweParams {
                log_q: 64,
                ..LweParams::current()
            },
        ] {
            let m = Manifest { lwe, ..sample() };
            assert!(m.validate().unwrap_err().0.contains("lattice parameters"));
        }
    }

    #[test]
    fn shapes_that_would_exhaust_a_client_are_rejected() {
        let mut m = sample();
        m.payload.cols = MAX_COLS + 1;
        assert!(m.validate().is_err());
        let mut m = sample();
        m.rank.rows = MAX_ROWS + 1;
        assert!(m.validate().is_err());
    }

    #[test]
    fn inconsistent_fields_are_rejected() {
        type Tamper = Box<dyn Fn(&mut Manifest)>;
        let cases: Vec<Tamper> = vec![
            Box::new(|m| m.format = "other".into()),
            Box::new(|m| m.dim = 9),
            Box::new(|m| m.rank.cols = 7),
            Box::new(|m| m.payload.log_p = 9),
            Box::new(|m| m.probes = 0),
            Box::new(|m| m.probes = 3),
            Box::new(|m| m.pages = 0),
            Box::new(|m| m.doc_scale = 0.0),
            Box::new(|m| m.doc_scale = f32::NAN),
            Box::new(|m| m.rank.log_p = 31),
            Box::new(|m| {
                m.blobs.remove("rank_hint");
            }),
        ];
        for (i, change) in cases.iter().enumerate() {
            let mut m = sample();
            change(&mut m);
            assert!(m.validate().is_err(), "case {i} was accepted");
        }
    }

    #[test]
    fn garbage_is_rejected_without_panicking() {
        assert!(Manifest::parse(b"").is_err());
        assert!(Manifest::parse(b"{\"format\":1}").is_err());
    }

    /// A server changing anything about the content changes the matrix it must use.
    #[test]
    fn the_public_matrix_seed_is_bound_to_the_content() {
        let base = sample();
        let mut a = vec![0u32; LWE_DIM];
        let mut b = vec![0u32; LWE_DIM];
        base.rank_matrix().row_into(0, &mut a);

        let mut other = sample();
        other.blobs.get_mut("layout").unwrap().sha256 = sha256_hex(b"different");
        other.rank_matrix().row_into(0, &mut b);
        assert_ne!(a, b);

        let mut other = sample();
        other.documents = 11;
        other.rank_matrix().row_into(0, &mut b);
        assert_ne!(a, b);

        base.payload_matrix().row_into(0, &mut b);
        assert_ne!(a, b, "the two databases must not share a matrix");
    }

    /// The manifest has no field for the seed, so a server cannot supply one.
    #[test]
    fn a_seed_cannot_be_supplied_by_the_server() {
        let mut value = serde_json::to_value(sample()).unwrap();
        value["seed"] = serde_json::json!("00");
        let parsed: Manifest = serde_json::from_value(value).unwrap();
        assert_eq!(parsed.content_id(), sample().content_id());
    }

    fn id(m: &Manifest) -> String {
        index_id(&serde_json::to_vec(m).unwrap())
    }

    #[test]
    fn hints_change_the_index_id_but_not_the_matrix() {
        let base = sample();
        let mut other = sample();
        other.blobs.get_mut("rank_hint").unwrap().sha256 = sha256_hex(b"different");
        assert_eq!(base.content_id(), other.content_id());
        assert_ne!(id(&base), id(&other));
    }

    /// Pinning an index must pin everything a server could otherwise vary per client, such as the
    /// traffic shape, which could serve as a tag or quietly weaken the results.
    #[test]
    fn every_field_is_covered_by_the_index_id() {
        let base = id(&sample());
        let changes: [fn(&mut Manifest); 6] = [
            |m| m.probes = 2,
            |m| m.pages = 2,
            |m| m.name = "other".into(),
            |m| m.rank.failure_log2 = -99.0,
            |m| m.payload.failure_log2 = -99.0,
            |m| m.blobs.get_mut("model").unwrap().bytes = 7,
        ];
        for change in changes {
            let mut other = sample();
            change(&mut other);
            assert_ne!(id(&other), base);
        }
    }

    /// A hash ends up in a cache path, so `/dev/zero` must not pass for one.
    #[test]
    fn a_blob_hash_must_be_a_hash() {
        for bad in ["/dev/zero", "../../etc/passwd", "ABC", "", &"A".repeat(64)] {
            let mut m = sample();
            m.blobs.get_mut("routing").unwrap().sha256 = bad.to_owned();
            assert!(m.validate().is_err(), "{bad}");
        }
    }

    #[test]
    fn sizes_that_would_overflow_32_bits_are_rejected() {
        let changes: [fn(&mut Manifest); 7] = [
            |m| m.rank.rows = 3_355_445,
            |m| m.payload.rows = 1 << 40,
            |m| m.model_dim = 1 << 61,
            |m| {
                m.model_dim = 1 << 13;
                m.dim = 1 << 13;
            },
            |m| m.prototypes = 1 << 62,
            |m| m.rank.cols = 1 << 40,
            |m| m.clusters = 1 << 62,
        ];
        for change in changes {
            let mut m = sample();
            change(&mut m);
            assert!(m.validate().is_err());
        }
    }

    #[test]
    fn file_sizes_must_follow_from_the_manifest() {
        for name in [
            "routing",
            "layout",
            "page_digests",
            "rank_hint",
            "payload_hint",
        ] {
            let mut m = sample();
            m.blobs.get_mut(name).unwrap().bytes += 1;
            assert!(m.validate().is_err(), "{name}");
        }
    }

    #[test]
    fn unusable_scales_are_rejected() {
        for scale in [0.0, -1.0, 1e-30, 1e30, f32::INFINITY, f32::NAN] {
            let mut m = sample();
            m.doc_scale = scale;
            assert!(m.validate().is_err(), "{scale}");
        }
    }

    #[test]
    fn work_per_search_is_bounded() {
        let mut m = sample();
        m.payload.cols = MAX_COLS;
        m.documents = MAX_COLS as u64;
        m.pages = 64;
        m.blobs.get_mut("layout").unwrap().bytes = m.expected_bytes("layout").unwrap();
        m.blobs.get_mut("page_digests").unwrap().bytes = m.expected_bytes("page_digests").unwrap();
        let error = m.validate().unwrap_err();
        assert!(error.0.contains("too much ciphertext"), "{error}");
    }
}
