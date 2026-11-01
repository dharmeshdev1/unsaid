//! The Unsaid private search protocol.
//!
//! An index is a set of public files that any client may download and two private matrices
//! that only the server holds. A search has two rounds. First the client sends encrypted
//! query vectors and gets back encrypted similarity scores for every document in the
//! clusters it chose. Then it picks the best documents and retrieves the pages holding them
//! with private information retrieval. The server computes both answers without being able
//! to tell which clusters, documents or pages were involved.
//!
//! This crate performs no I/O, so the same code runs on a server, on the command line and in
//! a browser.

pub mod client;
pub mod manifest;
pub mod pages;

pub use client::{
    FetchStage, Hit, Index, IndexFiles, Outcome, PickedStage, RankStage, ScoredStage, SearchError,
    SearchPlan,
};
pub use manifest::{BlobInfo, Manifest, ManifestError};
pub use pages::Document;
