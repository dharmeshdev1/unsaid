//! A BERT sentence encoder written from scratch in safe Rust.
//!
//! A private search client has to embed the query on the user's own device, in whatever
//! environment it runs in, and the result must agree with the embeddings the index was built
//! with. Having one implementation for the command line and for the browser gives that
//! directly, keeps the binary small, and removes native dependencies from the build.

mod bert;
mod model;
mod tokenizer;

pub use bert::{Bert, Pooling};
pub use model::{Embedder, ModelError, PackOptions, WeightFormat, pack_safetensors};
pub use tokenizer::Tokenizer;
