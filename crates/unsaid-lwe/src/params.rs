//! The one parameter set. There is deliberately no way to choose another.

/// LWE secret dimension `n`.
///
/// Published single-server PIR schemes commonly use `n = 1024` with these `q` and `sigma`.
/// That sits near 2^95 under the Core-SVP cost model. `n = 1280` is near 2^129, above the
/// 2^118 published for ML-KEM-512. See `docs/security.md` and `tools/lwe_estimate.py`.
pub const LWE_DIM: usize = 1280;

/// Ciphertext modulus is `2^LOG_Q`, so arithmetic is plain wrapping `u32`.
pub const LOG_Q: u32 = 32;

/// Standard deviation of the discrete Gaussian error.
pub const SIGMA: f64 = 6.4;

/// Default bound on the probability that one query decrypts incorrectly.
pub const MAX_FAILURE_LOG2: f64 = -40.0;
