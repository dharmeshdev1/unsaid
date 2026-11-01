//! The server's side of the scheme: a public random matrix, the private plaintext matrix,
//! and the two products that make everything work.
//!
//! `apply` is the only thing the server does to answer a query. It reads every entry of
//! the matrix for every query and never branches on a ciphertext value, so neither its
//! access pattern nor its running time depends on what was asked.

use crate::chacha::expand12;
use crate::params::LWE_DIM;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// The public LWE matrix `A`, with `cols` rows of [`LWE_DIM`] entries. It is expanded on
/// demand from a seed and never stored, because it can reach gigabytes.
#[derive(Clone, Debug)]
pub struct PublicMatrix {
    seed: [u8; 32],
    cols: usize,
}

impl PublicMatrix {
    /// `A` for a database with `cols` columns.
    pub fn new(seed: [u8; 32], cols: usize) -> Self {
        Self { seed, cols }
    }

    /// Number of LWE samples: one per database column.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Write row `j` of `A` into `out`, which must hold [`LWE_DIM`] words.
    #[inline]
    pub fn row_into(&self, j: usize, out: &mut [u32]) {
        debug_assert_eq!(out.len(), LWE_DIM);
        expand12(&self.seed, j as u64, out);
    }
}

/// Wrapping inner product of signed bytes with words.
#[inline]
fn dot_i8(row: &[i8], x: &[u32]) -> u32 {
    row.iter()
        .zip(x)
        .fold(0u32, |acc, (&m, &v)| acc.wrapping_add((m as i32 as u32).wrapping_mul(v)))
}

/// Wrapping inner product of two word vectors.
#[inline]
pub(crate) fn dot_u32(a: &[u32], b: &[u32]) -> u32 {
    a.iter().zip(b).fold(0u32, |acc, (&x, &y)| acc.wrapping_add(x.wrapping_mul(y)))
}

/// A database as a dense row-major matrix of signed bytes.
#[derive(Clone, Debug)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<i8>,
}

impl Matrix {
    /// Wrap row-major data.
    ///
    /// # Panics
    /// If `data.len() != rows * cols`.
    pub fn new(rows: usize, cols: usize, data: Vec<i8>) -> Self {
        assert_eq!(data.len(), rows * cols, "matrix data has the wrong length");
        Self { rows, cols, data }
    }

    /// Reinterpret raw bytes as a matrix, without copying semantics changing any value.
    pub fn from_bytes(rows: usize, cols: usize, bytes: &[u8]) -> Self {
        Self::new(rows, cols, bytes.iter().map(|&b| b as i8).collect())
    }

    /// Number of rows, which is the length of an answer.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns, which is the length of a query.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Row-major entries.
    pub fn data(&self) -> &[i8] {
        &self.data
    }

    /// Row `i`.
    pub fn row(&self, i: usize) -> &[i8] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// Largest squared Euclidean norm of any row. This decides the noise budget.
    pub fn max_row_norm_sq(&self) -> u64 {
        let norm = |row: &[i8]| row.iter().map(|&v| (i64::from(v) * i64::from(v)) as u64).sum();
        #[cfg(feature = "parallel")]
        let max = self.data.par_chunks(self.cols.max(1)).map(norm).max();
        #[cfg(not(feature = "parallel"))]
        let max = self.data.chunks(self.cols.max(1)).map(norm).max();
        max.unwrap_or(0)
    }

    /// Multiply this matrix by an encrypted vector. This is the server's whole job.
    ///
    /// # Panics
    /// If `x` does not have exactly `cols` entries. Callers validate request sizes first.
    pub fn apply(&self, x: &[u32]) -> Vec<u32> {
        assert_eq!(x.len(), self.cols, "ciphertext has the wrong length");
        let mut out = vec![0u32; self.rows];
        if self.cols == 0 {
            return out;
        }
        #[cfg(feature = "parallel")]
        out.par_iter_mut()
            .zip(self.data.par_chunks(self.cols))
            .for_each(|(o, row)| *o = dot_i8(row, x));
        #[cfg(not(feature = "parallel"))]
        for (o, row) in out.iter_mut().zip(self.data.chunks(self.cols)) {
            *o = dot_i8(row, x);
        }
        out
    }

    /// Precompute the hint `H = M * A`, stored row-major with [`LWE_DIM`] columns.
    ///
    /// The hint is what lets a client strip the mask from an answer without the server ever
    /// holding a key. It depends only on public `A` and on `M`, so it is the same for every
    /// client and is computed once.
    pub fn hint(&self, a: &PublicMatrix) -> Vec<u32> {
        assert_eq!(a.cols(), self.cols, "public matrix does not fit this database");
        let mut hint = vec![0u32; self.rows * LWE_DIM];
        // Working through blocks of A keeps the per-row loop inside the cache.
        const BLOCK: usize = 256;
        let mut a_block = vec![0u32; BLOCK * LWE_DIM];
        for start in (0..self.cols).step_by(BLOCK) {
            let end = (start + BLOCK).min(self.cols);
            let live = &mut a_block[..(end - start) * LWE_DIM];
            #[cfg(feature = "parallel")]
            live.par_chunks_mut(LWE_DIM)
                .enumerate()
                .for_each(|(k, row)| a.row_into(start + k, row));
            #[cfg(not(feature = "parallel"))]
            for (k, row) in live.chunks_mut(LWE_DIM).enumerate() {
                a.row_into(start + k, row);
            }
            let live = &*live;
            let accumulate = |(i, h): (usize, &mut [u32])| {
                let entries = &self.data[i * self.cols + start..i * self.cols + end];
                for (&m, a_row) in entries.iter().zip(live.chunks_exact(LWE_DIM)) {
                    if m != 0 {
                        let m = m as i32 as u32;
                        for (h_t, &a_t) in h.iter_mut().zip(a_row) {
                            *h_t = h_t.wrapping_add(m.wrapping_mul(a_t));
                        }
                    }
                }
            };
            #[cfg(feature = "parallel")]
            hint.par_chunks_mut(LWE_DIM).enumerate().for_each(accumulate);
            #[cfg(not(feature = "parallel"))]
            hint.chunks_mut(LWE_DIM).enumerate().for_each(accumulate);
        }
        hint
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_matrix(rows: usize, cols: usize, seed: u64) -> Matrix {
        let mut state = seed | 1;
        let data = (0..rows * cols)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as i8
            })
            .collect();
        Matrix::new(rows, cols, data)
    }

    #[test]
    fn rows_of_a_are_stable_and_distinct() {
        let a = PublicMatrix::new([5u8; 32], 10);
        let (mut r1, mut r2, mut r3) = (vec![0; LWE_DIM], vec![0; LWE_DIM], vec![0; LWE_DIM]);
        a.row_into(3, &mut r1);
        a.row_into(3, &mut r2);
        a.row_into(4, &mut r3);
        assert_eq!(r1, r2);
        assert_ne!(r1, r3);
    }

    #[test]
    fn apply_matches_a_naive_product_with_sign_extension() {
        let m = pseudo_matrix(13, 257, 42);
        let x: Vec<u32> = (0..257u32).map(|i| i.wrapping_mul(2654435761)).collect();
        let got = m.apply(&x);
        for i in 0..13 {
            let mut want = 0u32;
            for j in 0..257 {
                want = want.wrapping_add((i32::from(m.row(i)[j]) as u32).wrapping_mul(x[j]));
            }
            assert_eq!(got[i], want, "row {i}");
        }
    }

    #[test]
    fn hint_matches_a_naive_product_across_block_boundaries() {
        for cols in [1, 255, 256, 257, 700] {
            let m = pseudo_matrix(5, cols, 7 + cols as u64);
            let a = PublicMatrix::new([8u8; 32], cols);
            let hint = m.hint(&a);
            let mut want = vec![0u32; 5 * LWE_DIM];
            let mut row = vec![0u32; LWE_DIM];
            for j in 0..cols {
                a.row_into(j, &mut row);
                for i in 0..5 {
                    let v = i32::from(m.row(i)[j]) as u32;
                    for t in 0..LWE_DIM {
                        want[i * LWE_DIM + t] =
                            want[i * LWE_DIM + t].wrapping_add(v.wrapping_mul(row[t]));
                    }
                }
            }
            assert_eq!(hint, want, "cols = {cols}");
        }
    }

    #[test]
    fn row_norm_is_exact_including_the_most_negative_byte() {
        let m = Matrix::new(2, 3, vec![-128, 0, 1, 2, 2, 1]);
        assert_eq!(m.max_row_norm_sq(), 128 * 128 + 1);
    }

    #[test]
    fn from_bytes_is_a_bitwise_reinterpretation() {
        let m = Matrix::from_bytes(1, 3, &[0x00, 0x7f, 0xff]);
        assert_eq!(m.data(), &[0, 127, -1]);
    }

    #[test]
    #[should_panic(expected = "wrong length")]
    fn wrong_ciphertext_length_is_refused() {
        pseudo_matrix(2, 4, 1).apply(&[0; 3]);
    }

    #[test]
    #[should_panic(expected = "wrong length")]
    fn wrong_data_length_is_refused() {
        Matrix::new(2, 2, vec![0; 3]);
    }
}
