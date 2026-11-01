//! Small linear algebra helpers, including a principal component analysis.

use rayon::prelude::*;

/// Dot product in a form compilers vectorise, by accumulating in separate lanes.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
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

/// Scale a vector to unit length. A zero vector stays zero.
pub fn normalize(v: &mut [f32]) {
    let norm = dot(v, v).sqrt();
    if norm > 1e-12 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Mean and leading principal directions of a set of vectors.
pub struct Pca {
    /// The mean, with `input_dim` entries.
    pub mean: Vec<f32>,
    /// `output_dim` rows of `input_dim` entries, strongest direction first.
    pub components: Vec<f32>,
    /// Share of the variance kept, between 0 and 1.
    pub explained: f64,
}

impl Pca {
    /// Fit to `data`, using at most `max_samples` evenly spaced rows.
    ///
    /// # Panics
    /// If `output_dim` is zero or larger than `input_dim`, or `data` is empty.
    pub fn fit(data: &[f32], input_dim: usize, output_dim: usize, max_samples: usize) -> Self {
        assert!(
            output_dim >= 1 && output_dim <= input_dim,
            "bad output dimension"
        );
        let total = data.len() / input_dim;
        assert!(total > 0, "no data");
        let step = (total / max_samples.max(1)).max(1);
        let rows: Vec<&[f32]> = data.chunks_exact(input_dim).step_by(step).collect();
        let count = rows.len() as f64;

        let mut mean = vec![0f64; input_dim];
        for row in &rows {
            for (m, &v) in mean.iter_mut().zip(*row) {
                *m += f64::from(v);
            }
        }
        mean.iter_mut().for_each(|m| *m /= count);

        // Fixed-size chunks, added up in order, keep the result independent of thread count.
        let parts: Vec<Vec<f64>> = rows
            .par_chunks(256)
            .map(|chunk| {
                let mut c = vec![0f64; input_dim * input_dim];
                let mut centred = vec![0f64; input_dim];
                for row in chunk {
                    for ((o, &v), m) in centred.iter_mut().zip(*row).zip(&mean) {
                        *o = f64::from(v) - m;
                    }
                    for i in 0..input_dim {
                        let a = centred[i];
                        let target = &mut c[i * input_dim..(i + 1) * input_dim];
                        for (t, &b) in target[i..].iter_mut().zip(&centred[i..]) {
                            *t += a * b;
                        }
                    }
                }
                c
            })
            .collect();
        let mut cov = vec![0f64; input_dim * input_dim];
        for part in &parts {
            for (a, b) in cov.iter_mut().zip(part) {
                *a += b;
            }
        }
        for i in 0..input_dim {
            for j in i..input_dim {
                let v = cov[i * input_dim + j] / count;
                cov[i * input_dim + j] = v;
                cov[j * input_dim + i] = v;
            }
        }

        let (values, vectors) = symmetric_eigen(cov, input_dim);
        let mut order: Vec<usize> = (0..input_dim).collect();
        order.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
        let sum: f64 = values.iter().map(|v| v.max(0.0)).sum();
        let kept: f64 = order[..output_dim]
            .iter()
            .map(|&i| values[i].max(0.0))
            .sum();

        let mut components = Vec::with_capacity(output_dim * input_dim);
        for &column in &order[..output_dim] {
            let v: Vec<f64> = (0..input_dim)
                .map(|r| vectors[r * input_dim + column])
                .collect();
            // The sign of an eigenvector is arbitrary, so fix one for reproducibility.
            let pivot = v
                .iter()
                .fold(0f64, |best, &x| if x.abs() > best.abs() { x } else { best });
            let sign = if pivot < 0.0 { -1.0 } else { 1.0 };
            components.extend(v.iter().map(|&x| (x * sign) as f32));
        }
        Self {
            mean: mean.iter().map(|&m| m as f32).collect(),
            components,
            explained: if sum > 0.0 { kept / sum } else { 1.0 },
        }
    }

    /// Length of output vectors.
    pub fn output_dim(&self) -> usize {
        self.components.len() / self.mean.len()
    }

    /// Project one vector and normalise the result.
    pub fn project(&self, input: &[f32], output: &mut [f32]) {
        let n = self.mean.len();
        let centred: Vec<f32> = input.iter().zip(&self.mean).map(|(a, b)| a - b).collect();
        for (o, row) in output.iter_mut().zip(self.components.chunks_exact(n)) {
            *o = dot(row, &centred);
        }
        normalize(output);
    }
}

/// Eigen decomposition of a symmetric matrix by cyclic Jacobi rotations. Slow for huge
/// matrices but accurate and simple, and embedding sizes are small. Returns the eigenvalues
/// and a row-major matrix whose columns are the eigenvectors.
pub fn symmetric_eigen(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let scale: f64 = a
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
    for _sweep in 0..60 {
        let mut off = 0f64;
        for i in 0..n {
            for j in i + 1..n {
                off += a[i * n + j] * a[i * n + j];
            }
        }
        if off.sqrt() <= 1e-13 * scale {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum().max(if theta == 0.0 { 1.0 } else { -1.0 })
                    / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = (t * t + 1.0).sqrt().recip();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let values = (0..n).map(|i| a[i * n + i]).collect();
    (values, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_is_exact_on_small_integers_of_any_length() {
        for len in [0, 1, 7, 8, 9, 31] {
            let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let want: f32 = a.iter().map(|x| x * x).sum();
            assert_eq!(dot(&a, &a), want);
        }
    }

    #[test]
    fn eigen_decomposition_of_a_known_matrix() {
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3.
        let (mut values, _) = symmetric_eigen(vec![2.0, 1.0, 1.0, 2.0], 2);
        values.sort_by(f64::total_cmp);
        assert!((values[0] - 1.0).abs() < 1e-12 && (values[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn eigenvectors_reconstruct_the_matrix_and_are_orthonormal() {
        let n = 12;
        let mut a = vec![0f64; n * n];
        for i in 0..n {
            for j in 0..=i {
                let v = ((i * 7 + j * 13) % 11) as f64 - 5.0;
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
        let (values, v) = symmetric_eigen(a.clone(), n);
        for i in 0..n {
            for j in 0..n {
                let mut rebuilt = 0.0;
                let mut gram = 0.0;
                for k in 0..n {
                    rebuilt += v[i * n + k] * values[k] * v[j * n + k];
                    gram += v[k * n + i] * v[k * n + j];
                }
                assert!((rebuilt - a[i * n + j]).abs() < 1e-9);
                assert!((gram - f64::from(u8::from(i == j))).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn a_diagonal_matrix_terminates_immediately_with_its_diagonal() {
        let (values, _) = symmetric_eigen(vec![5.0, 0.0, 0.0, 7.0], 2);
        assert_eq!(values, vec![5.0, 7.0]);
    }

    #[test]
    fn pca_finds_the_direction_of_greatest_variance() {
        // Points along (1, 1, 0) with a little noise along (0, 0, 1).
        let mut data = Vec::new();
        for i in 0..200 {
            let t = i as f32 - 100.0;
            let noise = if i % 2 == 0 { 0.5 } else { -0.5 };
            data.extend_from_slice(&[t, t, noise]);
        }
        let pca = Pca::fit(&data, 3, 1, 1000);
        let c = &pca.components[..3];
        let half = std::f64::consts::FRAC_1_SQRT_2;
        assert!((f64::from(c[0].abs()) - half).abs() < 1e-3);
        assert!((f64::from(c[1].abs()) - half).abs() < 1e-3);
        assert!(c[2].abs() < 1e-3);
        assert!(pca.explained > 0.999);
    }

    #[test]
    fn pca_is_deterministic_and_independent_of_thread_count() {
        let data: Vec<f32> = (0..64 * 400)
            .map(|i| ((i * 2654435761u64 as usize) >> 7) as u8 as f32)
            .collect();
        let a = Pca::fit(&data, 64, 8, 1000);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let b = pool.install(|| Pca::fit(&data, 64, 8, 1000));
        assert_eq!(a.components, b.components);
        assert_eq!(a.mean, b.mean);
    }

    #[test]
    fn projection_outputs_unit_vectors() {
        let data: Vec<f32> = (0..30 * 50).map(|i| (i as f32 * 0.37).sin()).collect();
        let pca = Pca::fit(&data, 30, 5, 100);
        let mut out = vec![0f32; 5];
        pca.project(&data[..30], &mut out);
        assert!((dot(&out, &out) - 1.0).abs() < 1e-5);
    }
}
