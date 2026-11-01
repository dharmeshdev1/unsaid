//! The client's side of the scheme: fresh-secret linear encryption.
//!
//! A query vector `x` is encrypted as `b = A s + e + delta x`, where `s` is a new uniform
//! secret and `e` is small Gaussian noise. The server returns `M b`, which equals
//! `(M A) s + M e + delta M x`. The client knows the hint `H = M A` and its own `s`, so it
//! subtracts `H s` and rounds away `M e`, leaving `M x`. The server never sees `x`.
//!
//! The dangerous mistake with this scheme is using one mask for two plaintexts: subtracting
//! the ciphertexts reveals the difference of the plaintexts. [`Prepared::seal`] consumes
//! its value, so doing that does not compile.
//!
//! ```compile_fail
//! # use unsaid_lwe::{matrix::PublicMatrix, query::Prepared};
//! let a = PublicMatrix::new([0u8; 32], 16);
//! let token = Prepared::new(&a).unwrap();
//! let first = token.seal(8, &[(0, 1)]);
//! let second = token.seal(8, &[(1, 1)]); // error: use of moved value
//! ```

use zeroize::Zeroize;

use crate::chacha::Drbg;
use crate::gauss;
use crate::matrix::{PublicMatrix, dot_u32};
use crate::params::{LOG_Q, LWE_DIM};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// A query secret. Wiped from memory when dropped.
pub struct Secret(Vec<u32>);

impl Drop for Secret {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

/// A one-time encryption of the zero vector, ready to be turned into a ciphertext.
///
/// Computing `A s + e` is the expensive part of a query, because it expands the whole of
/// `A`. Doing it ahead of time means a search only has to add the plaintext.
pub struct Prepared {
    secret: Secret,
    mask: Vec<u32>,
}

impl Prepared {
    /// Prepare a token using fresh operating-system entropy.
    pub fn new(a: &PublicMatrix) -> Result<Self, getrandom::Error> {
        Ok(Self::with_rng(a, &mut Drbg::from_os()?))
    }

    /// Prepare a token with a caller-supplied generator. Only tests should use this:
    /// a predictable generator gives a predictable secret.
    pub fn with_rng(a: &PublicMatrix, rng: &mut Drbg) -> Self {
        let secret: Vec<u32> = (0..LWE_DIM).map(|_| rng.next_u32()).collect();
        // Draw the noise first and in order, so the result does not depend on threading.
        let mut mask: Vec<u32> = (0..a.cols()).map(|_| gauss::sample(rng) as u32).collect();
        let add_row = |(j, m): (usize, &mut u32), row: &mut Vec<u32>| {
            a.row_into(j, row);
            *m = m.wrapping_add(dot_u32(row, &secret));
        };
        #[cfg(feature = "parallel")]
        mask.par_iter_mut()
            .enumerate()
            .for_each_init(|| vec![0u32; LWE_DIM], |row, item| add_row(item, row));
        #[cfg(not(feature = "parallel"))]
        {
            let mut row = vec![0u32; LWE_DIM];
            for item in mask.iter_mut().enumerate() {
                add_row(item, &mut row);
            }
        }
        Self {
            secret: Secret(secret),
            mask,
        }
    }

    /// Number of entries the ciphertext will have.
    pub fn len(&self) -> usize {
        self.mask.len()
    }

    /// Whether the ciphertext would be empty.
    pub fn is_empty(&self) -> bool {
        self.mask.is_empty()
    }

    /// Encrypt a sparse plaintext vector in `Z_p`, with `p = 2^log_p`, using up this token.
    /// Entries not listed are zero. Values are taken modulo `p`.
    ///
    /// # Panics
    /// If `log_p` is outside `1..=31` or an index is out of range.
    pub fn seal(self, log_p: u32, plaintext: &[(usize, i32)]) -> (Secret, Vec<u32>) {
        let shift = LOG_Q - check_log_p(log_p);
        let Self { secret, mut mask } = self;
        for &(index, value) in plaintext {
            mask[index] = mask[index].wrapping_add((value as u32) << shift);
        }
        (secret, mask)
    }
}

fn check_log_p(log_p: u32) -> u32 {
    assert!(
        (1..LOG_Q).contains(&log_p),
        "plaintext modulus out of range"
    );
    log_p
}

impl Secret {
    /// Remove the mask from an answer. `hint` is row-major with [`LWE_DIM`] columns and has
    /// one row per answer entry. The result holds values in `0..p`.
    ///
    /// # Panics
    /// If the lengths disagree or `log_p` is outside `1..=31`.
    pub fn open(&self, log_p: u32, answer: &[u32], hint: &[u32]) -> Vec<u32> {
        let shift = LOG_Q - check_log_p(log_p);
        assert_eq!(
            Some(hint.len()),
            answer.len().checked_mul(LWE_DIM),
            "hint does not fit this answer"
        );
        let half = 1u32 << (shift - 1);
        answer
            .iter()
            .zip(hint.chunks_exact(LWE_DIM))
            .map(|(&ans, h)| ans.wrapping_sub(dot_u32(h, &self.0)).wrapping_add(half) >> shift)
            .collect()
    }

    /// Like [`Secret::open`], interpreting each value as a signed number in `-p/2..p/2`.
    pub fn open_signed(&self, log_p: u32, answer: &[u32], hint: &[u32]) -> Vec<i32> {
        let shift = LOG_Q - log_p;
        self.open(log_p, answer, hint)
            .into_iter()
            .map(|v| ((v << shift) as i32) >> shift)
            .collect()
    }

    /// Like [`Secret::open`], for a database of bytes. Requires `log_p == 8`.
    pub fn open_bytes(&self, answer: &[u32], hint: &[u32]) -> Vec<u8> {
        self.open(8, answer, hint)
            .into_iter()
            .map(|v| v as u8)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;

    fn matrix(rows: usize, cols: usize, seed: u64, bound: i64) -> Matrix {
        let mut state = seed | 1;
        let span = 2 * bound as u64 + 1;
        let data = (0..rows * cols)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state % span) as i64 - bound) as i8
            })
            .collect();
        Matrix::new(rows, cols, data)
    }

    #[test]
    fn decrypted_answers_equal_the_plaintext_product() {
        let (rows, cols) = (32, 512);
        let m = matrix(rows, cols, 11, 40);
        let a = PublicMatrix::new([1u8; 32], cols);
        let hint = m.hint(&a);
        let plain: Vec<(usize, i32)> = (0..16).map(|k| (100 + k, k as i32 - 8)).collect();
        let (secret, ct) = Prepared::new(&a).unwrap().seal(14, &plain);
        let got = secret.open_signed(14, &m.apply(&ct), &hint);
        for (i, &value) in got.iter().enumerate() {
            let want: i32 = plain.iter().map(|&(j, v)| i32::from(m.row(i)[j]) * v).sum();
            assert_eq!(value, want, "row {i}");
        }
    }

    #[test]
    fn a_unit_vector_retrieves_one_column_of_bytes_exactly() {
        let (rows, cols) = (64, 200);
        let bytes: Vec<u8> = (0..rows * cols).map(|i| (i * 37 % 251) as u8).collect();
        let m = Matrix::from_bytes(rows, cols, &bytes);
        let a = PublicMatrix::new([2u8; 32], cols);
        let hint = m.hint(&a);
        for column in [0, 1, 99, 199] {
            let (secret, ct) = Prepared::new(&a).unwrap().seal(8, &[(column, 1)]);
            let got = secret.open_bytes(&m.apply(&ct), &hint);
            let want: Vec<u8> = (0..rows).map(|r| bytes[r * cols + column]).collect();
            assert_eq!(got, want, "column {column}");
        }
    }

    #[test]
    fn signed_results_wrap_modulo_p_as_documented() {
        let m = Matrix::new(1, 1, vec![100]);
        let a = PublicMatrix::new([3u8; 32], 1);
        let hint = m.hint(&a);
        // 100 * 3 = 300 is outside -128..128, so it wraps to 300 - 256 = 44.
        let (secret, ct) = Prepared::new(&a).unwrap().seal(8, &[(0, 3)]);
        assert_eq!(secret.open_signed(8, &m.apply(&ct), &hint), vec![44]);
    }

    #[test]
    fn negative_plaintext_values_are_handled() {
        let m = Matrix::new(1, 2, vec![-5, 7]);
        let a = PublicMatrix::new([4u8; 32], 2);
        let hint = m.hint(&a);
        let (secret, ct) = Prepared::new(&a).unwrap().seal(12, &[(0, -9), (1, 2)]);
        assert_eq!(secret.open_signed(12, &m.apply(&ct), &hint), vec![45 + 14]);
    }

    #[test]
    fn the_same_query_never_produces_the_same_ciphertext() {
        let a = PublicMatrix::new([5u8; 32], 64);
        let (_, c1) = Prepared::new(&a).unwrap().seal(8, &[(3, 1)]);
        let (_, c2) = Prepared::new(&a).unwrap().seal(8, &[(3, 1)]);
        let equal = c1.iter().zip(&c2).filter(|(x, y)| x == y).count();
        assert!(
            equal < 3,
            "{equal} entries repeated between two encryptions"
        );
    }

    #[test]
    fn opening_with_the_wrong_secret_gives_noise_not_the_answer() {
        let (rows, cols) = (64, 64);
        let bytes = vec![65u8; rows * cols];
        let m = Matrix::from_bytes(rows, cols, &bytes);
        let a = PublicMatrix::new([6u8; 32], cols);
        let hint = m.hint(&a);
        let (_, ct) = Prepared::new(&a).unwrap().seal(8, &[(5, 1)]);
        let (other, _) = Prepared::new(&a).unwrap().seal(8, &[(5, 1)]);
        let got = other.open_bytes(&m.apply(&ct), &hint);
        let hits = got.iter().filter(|&&b| b == 65).count();
        assert!(hits < 8, "{hits} of {rows} bytes survived a wrong key");
    }

    #[test]
    fn with_rng_is_reproducible_regardless_of_threading() {
        let a = PublicMatrix::new([7u8; 32], 300);
        let run = |seed| {
            let mut rng = Drbg::from_seed(&[seed; 32]);
            Prepared::with_rng(&a, &mut rng).seal(8, &[]).1
        };
        assert_eq!(run(9), run(9));
        assert_ne!(run(9), run(10));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn a_plaintext_modulus_as_large_as_q_is_refused() {
        let a = PublicMatrix::new([8u8; 32], 2);
        let _ = Prepared::new(&a).unwrap().seal(32, &[]);
    }

    #[test]
    #[should_panic(expected = "hint does not fit")]
    fn a_mismatched_hint_is_refused() {
        let a = PublicMatrix::new([8u8; 32], 2);
        let (s, _) = Prepared::new(&a).unwrap().seal(8, &[]);
        s.open(8, &[1, 2], &[0; LWE_DIM]);
    }
}
