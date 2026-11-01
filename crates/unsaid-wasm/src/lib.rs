//! The Unsaid client for browsers: embedding, encryption and decryption inside the page.
//!
//! JavaScript handles the network and the interface. It only ever sees ciphertext and the
//! final results, and the query's secrets stay in WebAssembly memory where they are wiped
//! after use.

use std::collections::VecDeque;

use unsaid_core::Index;
use unsaid_core::client::{FetchStage, IndexFiles, RankStage, SearchPlan};
use unsaid_core::manifest::sha256_hex;
use unsaid_embed::Embedder;
use unsaid_lwe::{Matrix, Prepared};
use wasm_bindgen::prelude::*;

fn fail(error: impl std::fmt::Display) -> JsError {
    JsError::new(&error.to_string())
}

/// A connected search client.
#[wasm_bindgen]
pub struct Engine {
    index: Index,
    embedder: Embedder,
    plan: SearchPlan,
    rank_tokens: VecDeque<Prepared>,
    fetch_tokens: VecDeque<Prepared>,
    rank_stage: Option<RankStage>,
    fetch_stage: Option<FetchStage>,
}

#[wasm_bindgen]
impl Engine {
    /// Verify every public file against the manifest and load the model.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        manifest: &[u8],
        routing: &[u8],
        layout: &[u8],
        page_digests: &[u8],
        rank_hint: &[u8],
        payload_hint: &[u8],
        model: &[u8],
    ) -> Result<Engine, JsError> {
        let files = IndexFiles {
            routing,
            layout,
            page_digests,
            rank_hint,
            payload_hint,
        };
        let index = Index::open(manifest, &files).map_err(fail)?;
        let expected = index
            .manifest()
            .blobs
            .get("model")
            .map(|b| b.sha256.as_str());
        if expected != Some(sha256_hex(model).as_str()) {
            return Err(JsError::new("the model does not match the manifest"));
        }
        let embedder = Embedder::load(model).map_err(fail)?;
        let plan = index.default_plan();
        Ok(Self {
            index,
            embedder,
            plan,
            rank_tokens: VecDeque::new(),
            fetch_tokens: VecDeque::new(),
            rank_stage: None,
            fetch_stage: None,
        })
    }

    /// Display name of the index.
    pub fn name(&self) -> String {
        self.index.manifest().name.clone()
    }

    /// Number of documents in the index.
    pub fn documents(&self) -> f64 {
        self.index.manifest().documents as f64
    }

    /// Identifier of the index, which anyone can compare out of band.
    pub fn index_id(&self) -> String {
        self.index.manifest().index_id()
    }

    /// Searches that could start right now without waiting for encryption.
    pub fn prepared_searches(&self) -> usize {
        (self.rank_tokens.len() / self.plan.probes).min(self.fetch_tokens.len() / self.plan.pages)
    }

    /// Do the slow part of one future search ahead of time.
    pub fn prepare_one(&mut self) -> Result<(), JsError> {
        for _ in 0..self.plan.probes {
            self.rank_tokens
                .push_back(self.index.prepare_rank().map_err(fail)?);
        }
        for _ in 0..self.plan.pages {
            self.fetch_tokens
                .push_back(self.index.prepare_fetch().map_err(fail)?);
        }
        Ok(())
    }

    /// Embed a query on this device. Exposed separately so its cost can be shown.
    pub fn embed(&self, query: &str) -> Vec<f32> {
        self.embedder.embed_query(query)
    }

    /// Start a search from an embedding. Returns the bytes to send to the ranking endpoint.
    pub fn begin(&mut self, embedding: &[f32]) -> Result<Vec<u8>, JsError> {
        if self.prepared_searches() == 0 {
            self.prepare_one()?;
        }
        let tokens: Vec<Prepared> = self.rank_tokens.drain(..self.plan.probes).collect();
        let (stage, request) = self.index.begin(embedding, tokens).map_err(fail)?;
        self.rank_stage = Some(stage);
        Ok(request)
    }

    /// Decrypt the scores. Returns the bytes to send to the document endpoint.
    pub fn receive_rank(&mut self, answer: &[u8], limit: usize) -> Result<Vec<u8>, JsError> {
        let stage = self
            .rank_stage
            .take()
            .ok_or_else(|| JsError::new("no search in progress"))?;
        let tokens: Vec<Prepared> = self.fetch_tokens.drain(..self.plan.pages).collect();
        let (stage, request) = stage
            .receive(&self.index, answer, tokens, limit)
            .map_err(fail)?;
        self.fetch_stage = Some(stage);
        Ok(request)
    }

    /// Decrypt and check the documents. Returns the results as JSON.
    pub fn receive_pages(&mut self, answer: &[u8]) -> Result<String, JsError> {
        let stage = self
            .fetch_stage
            .take()
            .ok_or_else(|| JsError::new("no search in progress"))?;
        let outcome = stage.receive(&self.index, answer).map_err(fail)?;
        serde_json::to_string(&outcome).map_err(fail)
    }

    /// Clusters scored per search.
    pub fn plan_probes(&self) -> usize {
        self.plan.probes
    }

    /// Pages retrieved per search.
    pub fn plan_pages(&self) -> usize {
        self.plan.pages
    }
}

/// The server's arithmetic, for a demo page with no backend.
#[wasm_bindgen]
pub struct LocalServer {
    rank: Matrix,
    payload: Matrix,
}

#[wasm_bindgen]
impl LocalServer {
    /// Load the two matrices, each stored one byte per entry.
    #[wasm_bindgen(constructor)]
    pub fn new(
        rank_rows: usize,
        rank_cols: usize,
        rank: &[u8],
        payload_rows: usize,
        payload_cols: usize,
        payload: &[u8],
    ) -> Result<LocalServer, JsError> {
        if rank.len() != rank_rows * rank_cols || payload.len() != payload_rows * payload_cols {
            return Err(JsError::new("a matrix file has the wrong size"));
        }
        Ok(Self {
            rank: Matrix::from_bytes(rank_rows, rank_cols, rank),
            payload: Matrix::from_bytes(payload_rows, payload_cols, payload),
        })
    }

    /// Answer a ranking request.
    pub fn rank(&self, request: &[u8]) -> Result<Vec<u8>, JsError> {
        self.rank
            .answer(request)
            .ok_or_else(|| JsError::new("bad request size"))
    }

    /// Answer a document request.
    pub fn fetch(&self, request: &[u8]) -> Result<Vec<u8>, JsError> {
        self.payload
            .answer(request)
            .ok_or_else(|| JsError::new("bad request size"))
    }
}
