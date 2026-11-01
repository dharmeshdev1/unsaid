//! Build an Unsaid index from documents and their embeddings.
//!
//! Builds are deterministic. The same documents, embeddings and options produce byte-identical
//! files on any machine with any number of threads, so anyone holding a public corpus can
//! rebuild its index and compare the index id with what a server publishes.

pub mod cluster;
pub mod index;
pub mod ingest;
pub mod linalg;

pub use index::{BuildOptions, BuildReport, build};
pub use ingest::{IngestOptions, read_documents};
