//! Learning-with-errors primitives for private retrieval.
//!
//! This crate implements the linear encryption behind single-server private information
//! retrieval in the style of SimplePIR (Henzinger, Hong, Corrigan-Gibbs, Meiklejohn and
//! Vaikuntanathan, USENIX Security 2023), with a more conservative lattice dimension. A
//! client encrypts a vector, a server multiplies its plaintext matrix by the ciphertext,
//! and the client decrypts the product. The server learns nothing about the vector under
//! the LWE assumption, which is also what post-quantum key exchange relies on.

pub mod chacha;
pub mod gauss;
mod gauss_table;
pub mod matrix;
pub mod noise;
pub mod params;
pub mod query;

pub use matrix::{Matrix, PublicMatrix};
pub use params::{LOG_Q, LWE_DIM, SIGMA};
pub use query::{Prepared, Secret};
