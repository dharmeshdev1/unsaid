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

/// Longest ciphertext a client will agree to create, in entries. This bounds how much work
/// and memory a server can make a client spend.
pub const MAX_COLS: usize = 1 << 26;
/// Largest answer a client will agree to process, in entries.
pub const MAX_ROWS: usize = 1 << 22;

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
        if self.dim == 0 || self.model_dim == 0 || self.dim > self.model_dim {
            return fail(format!(
                "bad dimensions {} and {}",
                self.dim, self.model_dim
            ));
        }
        if self.clusters == 0 || self.prototypes == 0 {
            return fail("an index needs clusters and prototypes".into());
        }
        if self.probes == 0 || self.probes > self.clusters.min(64) {
            return fail(format!("bad probe count {}", self.probes));
        }
        if self.pages == 0 || self.pages > 64 {
            return fail(format!("bad page count {}", self.pages));
        }
        if !(self.doc_scale > 0.0 && self.query_scale > 0.0) {
            return fail("quantisation scales must be positive".into());
        }
        if self.rank.cols != self.clusters.saturating_mul(self.dim) {
            return fail("ranking matrix does not have clusters * dim columns".into());
        }
        if self.payload.log_p != 8 {
            return fail("the document matrix must hold bytes".into());
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
        for name in CONTENT_BLOBS.iter().chain(&HINT_BLOBS) {
            if !self.blobs.contains_key(*name) {
                return fail(format!("the blob {name} is missing"));
            }
        }
        Ok(())
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

    /// A short identifier covering the whole index, hints included. Publish this out of band
    /// and clients can detect a server that swaps the index for another one.
    pub fn index_id(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"unsaid/v1/index\0");
        h.update(self.content_id());
        for name in HINT_BLOBS {
            h.update(b"\0");
            if let Some(blob) = self.blobs.get(name) {
                h.update(blob.sha256.as_bytes());
            }
        }
        hex(&h.finalize()[..16])
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
        Manifest {
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
        }
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

    #[test]
    fn hints_change_the_index_id_but_not_the_matrix() {
        let base = sample();
        let mut other = sample();
        other.blobs.get_mut("rank_hint").unwrap().sha256 = sha256_hex(b"different");
        assert_eq!(base.content_id(), other.content_id());
        assert_ne!(base.index_id(), other.index_id());
    }
}
