//! The Unsaid client for browsers: embedding, encryption and decryption inside the page.
//!
//! JavaScript handles the network and the interface. It only ever sees ciphertext and the
//! final results, and the query's secrets stay in WebAssembly memory where they are wiped
//! after use.

use std::collections::VecDeque;

use unsaid_core::Index;
use unsaid_core::client::{
    FetchStage, IndexFiles, PickedStage, RankStage, ScoredStage, SearchPlan,
};
use unsaid_core::manifest::{Manifest, sha256_hex};
use unsaid_embed::{Embedder, is_known_model};
use unsaid_lwe::{Matrix, Prepared};
use wasm_bindgen::prelude::*;

fn fail(error: impl std::fmt::Display) -> JsError {
    JsError::new(&error.to_string())
}

/// Refuse a bad manifest before anything sized by it is allocated or downloaded.
#[wasm_bindgen]
pub fn check_manifest(manifest: &[u8]) -> Result<(), JsError> {
    Manifest::parse(manifest).map(|_| ()).map_err(fail)
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
    scored_stage: Option<ScoredStage>,
    picked_stage: Option<PickedStage>,
    fetch_stage: Option<FetchStage>,
}

#[wasm_bindgen]
impl Engine {
    /// Verify every public file against the manifest and load the model. Unless
    /// `allow_unknown_model` is set, the model must be on the trusted list. A page may set it when
    /// the server supplying the model also supplied the page, which already had to be trusted.
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
        allow_unknown_model: bool,
    ) -> Result<Engine, JsError> {
        let files = IndexFiles {
            routing,
            layout,
            page_digests,
            rank_hint,
            payload_hint,
        };
        let index = Index::open(manifest, &files).map_err(fail)?;
        let hash = sha256_hex(model);
        if index.manifest().blobs["model"].sha256 != hash {
            return Err(JsError::new("the model does not match the manifest"));
        }
        if !allow_unknown_model && !is_known_model(&hash) {
            return Err(JsError::new(
                "this index uses a model this page does not trust",
            ));
        }
        let embedder = Embedder::load(model).map_err(fail)?;
        if embedder.dim() != index.manifest().model_dim {
            return Err(JsError::new("the model's vectors do not fit this index"));
        }
        let plan = index.default_plan();
        Ok(Self {
            index,
            embedder,
            plan,
            rank_tokens: VecDeque::new(),
            fetch_tokens: VecDeque::new(),
            rank_stage: None,
            scored_stage: None,
            picked_stage: None,
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
        self.index.id().to_owned()
    }

    /// Searches that could start right now without waiting for encryption.
    pub fn prepared_searches(&self) -> usize {
        (self.rank_tokens.len() / self.plan.probes).min(self.fetch_tokens.len() / self.plan.pages)
    }

    /// Do the slow part of one future search ahead of time.
    pub fn prepare_one(&mut self) -> Result<(), JsError> {
        // A search that failed halfway used up only one kind of token, so top up each separately.
        let searches = self.prepared_searches() + 1;
        while self.rank_tokens.len() < searches * self.plan.probes {
            self.rank_tokens
                .push_back(self.index.prepare_rank().map_err(fail)?);
        }
        while self.fetch_tokens.len() < searches * self.plan.pages {
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

    /// Decrypt the scores. This takes the same time for every search, so its duration can be used
    /// to hide how long `pick_pages` takes.
    pub fn decrypt_rank(&mut self, answer: &[u8]) -> Result<(), JsError> {
        let stage = self
            .rank_stage
            .take()
            .ok_or_else(|| JsError::new("no search in progress"))?;
        self.scored_stage = Some(stage.decrypt(&self.index, answer).map_err(fail)?);
        Ok(())
    }

    /// Choose the pages to fetch. How long this takes depends on the results.
    pub fn pick_pages(&mut self, limit: usize) -> Result<(), JsError> {
        let stage = self
            .scored_stage
            .take()
            .ok_or_else(|| JsError::new("no search in progress"))?;
        self.picked_stage = Some(stage.pick(&self.index, self.plan.pages, limit));
        Ok(())
    }

    /// Returns the bytes to send to the document endpoint.
    pub fn seal_pages(&mut self) -> Result<Vec<u8>, JsError> {
        let stage = self
            .picked_stage
            .take()
            .ok_or_else(|| JsError::new("no search in progress"))?;
        let tokens: Vec<Prepared> = self.fetch_tokens.drain(..self.plan.pages).collect();
        let (stage, request) = stage.seal(&self.index, tokens).map_err(fail)?;
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

    /// Forget a search that will not finish, wiping its secrets.
    pub fn abort(&mut self) {
        self.rank_stage = None;
        self.scored_stage = None;
        self.picked_stage = None;
        self.fetch_stage = None;
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
