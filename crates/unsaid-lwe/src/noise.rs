//! The noise budget, computed from the actual database rather than assumed.
//!
//! Decryption is correct as long as the noise in an answer is under half a plaintext step.
//! That noise is `M e`, a sum of independent Gaussians weighted by a row of `M`, so it
//! behaves like a Gaussian with standard deviation `sigma * ||row||`. Summing the tail over
//! every row gives an explicit bound on the chance that one answer has any wrong entry.

use crate::params::{LOG_Q, SIGMA};

/// Base-2 logarithm of an upper bound on the probability that an answer with `rows` entries
/// contains at least one decryption error, for a matrix whose largest squared row norm is
/// `max_row_norm_sq`.
pub fn failure_log2(log_p: u32, max_row_norm_sq: u64, rows: usize) -> f64 {
    assert!(
        (1..LOG_Q).contains(&log_p),
        "plaintext modulus out of range"
    );
    if max_row_norm_sq == 0 || rows == 0 {
        return f64::NEG_INFINITY;
    }
    let half_step = (2f64).powi((LOG_Q - log_p - 1) as i32);
    let variance = SIGMA * SIGMA * max_row_norm_sq as f64;
    // P[|noise| >= t] <= 2 exp(-t^2 / (2 v)), with a union bound over the rows.
    let exponent = half_step * half_step / (2.0 * variance);
    1.0 + (rows as f64).log2() - exponent / std::f64::consts::LN_2
}

/// Largest plaintext modulus exponent whose failure bound is at most `2^max_failure_log2`.
/// `None` if not even `p = 2` qualifies.
pub fn largest_log_p(max_row_norm_sq: u64, rows: usize, max_failure_log2: f64) -> Option<u32> {
    (1..LOG_Q)
        .rev()
        .find(|&log_p| failure_log2(log_p, max_row_norm_sq, rows) <= max_failure_log2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::{Matrix, PublicMatrix};
    use crate::query::Prepared;

    #[test]
    fn a_bigger_modulus_and_a_bigger_norm_both_cost_budget() {
        assert!(failure_log2(14, 1 << 20, 100) > failure_log2(12, 1 << 20, 100));
        assert!(failure_log2(12, 1 << 22, 100) > failure_log2(12, 1 << 20, 100));
        assert!(failure_log2(12, 1 << 20, 1000) > failure_log2(12, 1 << 20, 100));
    }

    /// With a half step of 2^17 and a variance of 40.96 * 2^20, the exponent is exactly 200.
    #[test]
    fn the_bound_matches_a_calculation_by_hand() {
        let expected = 1.0 + 1000f64.log2() - 200.0 / std::f64::consts::LN_2;
        assert!((failure_log2(14, 1 << 20, 1000) - expected).abs() < 1e-9);
    }

    #[test]
    fn an_all_zero_matrix_cannot_fail() {
        assert_eq!(failure_log2(8, 0, 10), f64::NEG_INFINITY);
    }

    #[test]
    fn largest_log_p_is_on_the_boundary() {
        let (norm, rows) = (1u64 << 24, 2048usize);
        let p = largest_log_p(norm, rows, -40.0).unwrap();
        assert!(failure_log2(p, norm, rows) <= -40.0);
        assert!(failure_log2(p + 1, norm, rows) > -40.0);
    }

    #[test]
    fn an_impossible_budget_is_reported() {
        assert_eq!(largest_log_p(u64::MAX / 2, 1 << 20, -40.0), None);
    }

    /// The bound must hold against reality: measure the noise of real answers.
    #[test]
    fn measured_noise_matches_the_prediction() {
        let (rows, cols) = (64usize, 4096usize);
        let mut state = 88172645463325252u64;
        let data: Vec<i8> = (0..rows * cols)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state % 201) as i64 - 100) as i8
            })
            .collect();
        let m = Matrix::new(rows, cols, data);
        let a = PublicMatrix::new([9u8; 32], cols);
        let hint = m.hint(&a);
        let norm = m.max_row_norm_sq();

        // Encrypt zero, so that after the hint is removed only the noise is left.
        let mut samples = Vec::new();
        for _ in 0..12 {
            let (secret, ct) = Prepared::new(&a).unwrap().seal(31, &[]);
            let ans = m.apply(&ct);
            // log_p = 31 leaves one bit of rounding, so recover raw residues instead.
            samples.extend(residues(&secret, &ans, &hint));
        }
        let var = samples
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            / samples.len() as f64;
        let predicted = SIGMA * SIGMA * norm as f64;
        let ratio = var / predicted;
        // Rows are close to the maximum norm, and the sample has 768 entries.
        assert!(
            (0.75..1.25).contains(&ratio),
            "measured / predicted = {ratio:.3}"
        );
        // And no entry anywhere near the half step for p = 2^14.
        let half_step = 2f64.powi(32 - 14 - 1);
        assert!(samples.iter().all(|&x| f64::from(x).abs() < half_step));
    }

    fn residues(secret: &crate::query::Secret, ans: &[u32], hint: &[u32]) -> Vec<i32> {
        // Opening with two different moduli reconstructs the residue without exposing the
        // secret: value modulo 2^31 at rounding precision is the residue divided by 2.
        secret
            .open(31, ans, hint)
            .into_iter()
            .map(|v| (v << 1) as i32)
            .collect()
    }
}
