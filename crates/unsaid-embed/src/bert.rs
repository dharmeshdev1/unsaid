//! The BERT encoder forward pass.
//!
//! Matrices are stored so that the innermost loops are `y[i] += a * x[i]` over fixed-size
//! rows. Compilers vectorise that form without `unsafe` or fast-math, on every target
//! including WebAssembly, while they refuse to vectorise an ordinary floating point dot
//! product because that would reorder a sum.

use serde::{Deserialize, Serialize};

/// How token vectors become one sentence vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Pooling {
    /// Use the vector of the leading `[CLS]` token.
    Cls,
    /// Average all tokens.
    Mean,
}

/// A dense layer. `weight` is stored transposed: `inputs` rows of `outputs` values.
pub(crate) struct Linear {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub inputs: usize,
    pub outputs: usize,
}

impl Linear {
    /// Apply to `tokens` row-major input rows, overwriting `y`.
    fn forward(&self, x: &[f32], y: &mut [f32]) {
        y.copy_from_slice(&self.bias.repeat(y.len() / self.outputs));
        let n = self.outputs;
        for (x_row, y_row) in x.chunks_exact(self.inputs).zip(y.chunks_exact_mut(n)) {
            // Four weight rows at a time touch each output once per four multiplies, which
            // nearly doubles throughput compared with one row at a time.
            let mut xs = x_row.chunks_exact(4);
            let mut ws = self.weight.chunks_exact(4 * n);
            for (a, w) in (&mut xs).zip(&mut ws) {
                let (w0, rest) = w.split_at(n);
                let (w1, rest) = rest.split_at(n);
                let (w2, w3) = rest.split_at(n);
                let (a0, a1, a2, a3) = (a[0], a[1], a[2], a[3]);
                for i in 0..n {
                    y_row[i] += a0 * w0[i] + a1 * w1[i] + a2 * w2[i] + a3 * w3[i];
                }
            }
            for (&a, w_row) in xs.remainder().iter().zip(ws.remainder().chunks_exact(n)) {
                axpy(a, w_row, y_row);
            }
        }
    }
}

pub(crate) struct LayerNorm {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
}

impl LayerNorm {
    const EPS: f32 = 1e-12;

    fn apply(&self, x: &mut [f32]) {
        let dim = self.gamma.len();
        for row in x.chunks_exact_mut(dim) {
            let mean = sum(row) / dim as f32;
            let mut var = 0.0f32;
            for chunk in row.chunks(16) {
                let mut acc = 0.0f32;
                for &v in chunk {
                    acc += (v - mean) * (v - mean);
                }
                var += acc;
            }
            var /= dim as f32;
            let scale = (var + Self::EPS).sqrt().recip();
            for ((v, &g), &b) in row.iter_mut().zip(&self.gamma).zip(&self.beta) {
                *v = (*v - mean) * scale * g + b;
            }
        }
    }
}

pub(crate) struct Layer {
    pub query: Linear,
    pub key: Linear,
    pub value: Linear,
    pub attn_out: Linear,
    pub attn_norm: LayerNorm,
    pub ffn_in: Linear,
    pub ffn_out: Linear,
    pub ffn_norm: LayerNorm,
}

/// BERT without its pretraining heads.
pub struct Bert {
    pub(crate) hidden: usize,
    pub(crate) heads: usize,
    pub(crate) word: Vec<f32>,
    pub(crate) position: Vec<f32>,
    pub(crate) token_type: Vec<f32>,
    pub(crate) embed_norm: LayerNorm,
    pub(crate) layers: Vec<Layer>,
    pub(crate) pooling: Pooling,
}

#[inline(always)]
fn axpy(a: f32, x: &[f32], y: &mut [f32]) {
    for (y_i, &x_i) in y.iter_mut().zip(x) {
        *y_i += a * x_i;
    }
}

#[inline]
fn sum(x: &[f32]) -> f32 {
    x.iter().sum()
}

/// `erf` by Abramowitz and Stegun 7.1.28, error under 3e-7, in a form that vectorises.
#[inline(always)]
fn erf_abs(x: f32) -> f32 {
    let t = 1.0
        + x * (0.070_523_08
            + x * (0.042_282_01
                + x * (0.009_270_527
                    + x * (0.000_152_014_3 + x * (0.000_276_567_2 + x * 0.000_043_063_8)))));
    let t2 = t * t;
    let t4 = t2 * t2;
    let t8 = t4 * t4;
    1.0 - (t8 * t8).recip()
}

/// The exact GELU, `x * Phi(x)`, as PyTorch computes it by default.
#[inline]
fn gelu(x: &mut [f32]) {
    const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    for v in x.iter_mut() {
        let z = *v * FRAC_1_SQRT_2;
        let e = erf_abs(z.abs()).copysign(z);
        *v *= 0.5 * (1.0 + e);
    }
}

/// Dot product in a form that vectorises, by accumulating in eight separate lanes.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0f32; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for i in 0..8 {
            lanes[i] += x[i] * y[i];
        }
    }
    let mut total: f32 = lanes.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        total += x * y;
    }
    total
}

impl Bert {
    /// Embedding dimension.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Encode token ids into one unit-length sentence vector.
    ///
    /// # Panics
    /// If an id is outside the vocabulary or there are more tokens than position embeddings.
    pub fn encode(&self, ids: &[u32]) -> Vec<f32> {
        let h = self.hidden;
        let t = ids.len();
        let mut x = vec![0f32; t * h];
        for (pos, (row, &id)) in x.chunks_exact_mut(h).zip(ids).enumerate() {
            let word = &self.word[id as usize * h..(id as usize + 1) * h];
            let place = &self.position[pos * h..(pos + 1) * h];
            for i in 0..h {
                row[i] = word[i] + place[i] + self.token_type[i];
            }
        }
        self.embed_norm.apply(&mut x);

        let head_dim = h / self.heads;
        let scale = (head_dim as f32).sqrt().recip();
        let (mut q, mut k, mut v) = (vec![0f32; t * h], vec![0f32; t * h], vec![0f32; t * h]);
        let (mut ctx, mut out) = (vec![0f32; t * h], vec![0f32; t * h]);
        let mut weights = vec![0f32; t];
        for layer in &self.layers {
            layer.query.forward(&x, &mut q);
            layer.key.forward(&x, &mut k);
            layer.value.forward(&x, &mut v);
            ctx.fill(0.0);
            for head in 0..self.heads {
                let span = head * head_dim..(head + 1) * head_dim;
                for i in 0..t {
                    let q_i = &q[i * h..(i + 1) * h][span.clone()];
                    let mut best = f32::NEG_INFINITY;
                    for (j, w) in weights.iter_mut().enumerate() {
                        *w = dot(q_i, &k[j * h..(j + 1) * h][span.clone()]) * scale;
                        best = best.max(*w);
                    }
                    let mut total = 0.0f32;
                    for w in weights.iter_mut() {
                        *w = (*w - best).exp();
                        total += *w;
                    }
                    let c_i = &mut ctx[i * h..(i + 1) * h][span.clone()];
                    for (j, &w) in weights.iter().enumerate() {
                        axpy(w / total, &v[j * h..(j + 1) * h][span.clone()], c_i);
                    }
                }
            }
            layer.attn_out.forward(&ctx, &mut out);
            for (a, b) in x.iter_mut().zip(&out) {
                *a += *b;
            }
            layer.attn_norm.apply(&mut x);

            let mut wide = vec![0f32; t * layer.ffn_in.outputs];
            layer.ffn_in.forward(&x, &mut wide);
            gelu(&mut wide);
            layer.ffn_out.forward(&wide, &mut out);
            for (a, b) in x.iter_mut().zip(&out) {
                *a += *b;
            }
            layer.ffn_norm.apply(&mut x);
        }

        let mut pooled = match self.pooling {
            Pooling::Cls => x[..h].to_vec(),
            Pooling::Mean => {
                let mut acc = vec![0f32; h];
                for row in x.chunks_exact(h) {
                    axpy(1.0, row, &mut acc);
                }
                acc
            }
        };
        let norm = dot(&pooled, &pooled).sqrt().max(1e-12);
        for v in pooled.iter_mut() {
            *v /= norm;
        }
        pooled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_is_accurate_against_known_values() {
        // Values of erf from a high precision source.
        let known = [
            (0.1f32, 0.112_462_92f32),
            (0.5, 0.520_499_9),
            (1.0, 0.842_700_8),
            (1.5, 0.966_105_1),
            (2.0, 0.995_322_3),
            (3.0, 0.999_977_9),
        ];
        for (x, want) in known {
            let mut value = [x * std::f32::consts::SQRT_2];
            gelu(&mut value);
            let phi = value[0] / (x * std::f32::consts::SQRT_2);
            let got = 2.0 * phi - 1.0;
            assert!((got - want).abs() < 2e-6, "erf({x}) = {got}, want {want}");
        }
    }

    #[test]
    fn gelu_is_negative_small_on_the_left_and_near_identity_on_the_right() {
        let mut x = [-10.0f32, -1.0, 0.0, 1.0, 10.0];
        gelu(&mut x);
        assert!(x[0].abs() < 1e-5);
        assert!((x[1] - (-0.158_655_25)).abs() < 1e-5);
        assert_eq!(x[2], 0.0);
        assert!((x[3] - 0.841_344_75).abs() < 1e-5);
        assert!((x[4] - 10.0).abs() < 1e-4);
    }

    #[test]
    fn dot_handles_remainders() {
        let a: Vec<f32> = (0..19).map(|i| i as f32).collect();
        let want: f32 = a.iter().map(|v| v * v).sum();
        assert_eq!(dot(&a, &a), want);
    }

    #[test]
    fn linear_layer_matches_a_hand_computation() {
        // y = W x + b with W = [[1, 2, 3], [4, 5, 6]], so transposed storage is 1 4 2 5 3 6.
        let l = Linear {
            weight: vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
            bias: vec![0.5, -0.5],
            inputs: 3,
            outputs: 2,
        };
        let x = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let mut y = [9.0; 4];
        l.forward(&x, &mut y);
        assert_eq!(y, [6.5, 14.5, 2.5, 4.5]);
    }

    #[test]
    fn layer_norm_produces_zero_mean_unit_variance_before_scaling() {
        let n = LayerNorm { gamma: vec![1.0; 4], beta: vec![0.0; 4] };
        let mut x = [1.0, 2.0, 3.0, 4.0];
        n.apply(&mut x);
        let mean: f32 = x.iter().sum::<f32>() / 4.0;
        let var: f32 = x.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-6 && (var - 1.0).abs() < 1e-5);
    }
}
