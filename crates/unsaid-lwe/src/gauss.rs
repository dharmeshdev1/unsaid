//! Discrete Gaussian error sampling.
//!
//! The error hides the secret, so the sampler reads the whole table for every sample and
//! never branches on a secret value. That keeps its running time independent of the noise
//! it produces.

use crate::chacha::Drbg;
use crate::gauss_table::{CDT, TAIL};

/// One sample from the discrete Gaussian with `sigma = 6.4`.
#[inline]
pub fn sample(rng: &mut Drbg) -> i32 {
    let r = rng.next_u64();
    let mut index = 0u32;
    for &threshold in CDT.iter() {
        index += u32::from(r >= threshold);
    }
    index as i32 - TAIL
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::SIGMA;

    fn draws(n: usize, seed: u8) -> Vec<i32> {
        let mut rng = Drbg::from_seed(&[seed; 32]);
        (0..n).map(|_| sample(&mut rng)).collect()
    }

    #[test]
    fn moments_match_the_target() {
        let n = 2_000_000usize;
        let xs = draws(n, 1);
        let mean = xs.iter().map(|&x| f64::from(x)).sum::<f64>() / n as f64;
        let var = xs.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / n as f64;
        // Standard errors: sigma / sqrt(n) for the mean, sqrt(2) sigma^2 / sqrt(n) for the variance.
        assert!(mean.abs() < 5.0 * SIGMA / (n as f64).sqrt(), "mean {mean}");
        let var_err = 2f64.sqrt() * SIGMA * SIGMA / (n as f64).sqrt();
        assert!((var - SIGMA * SIGMA).abs() < 5.0 * var_err, "variance {var}");
    }

    #[test]
    fn distribution_fits_the_exact_probabilities() {
        let n = 2_000_000usize;
        let xs = draws(n, 2);
        let weight = |x: i32| (-(f64::from(x) * f64::from(x)) / (2.0 * SIGMA * SIGMA)).exp();
        let total: f64 = (-TAIL..=TAIL).map(weight).sum();
        let mut counts = vec![0u64; (2 * TAIL + 1) as usize];
        for &x in &xs {
            counts[(x + TAIL) as usize] += 1;
        }
        // Chi-square over the cells expected to hold at least 10 samples.
        let mut chi = 0.0;
        let mut cells = 0;
        for x in -TAIL..=TAIL {
            let expect = weight(x) / total * n as f64;
            if expect >= 10.0 {
                let got = counts[(x + TAIL) as usize] as f64;
                chi += (got - expect).powi(2) / expect;
                cells += 1;
            }
        }
        // Wilson-Hilferty bound at p = 1e-6 for cells - 1 degrees of freedom.
        let df = f64::from(cells - 1);
        let a = 2.0 / (9.0 * df);
        let limit = df * (1.0 - a + 4.75 * a.sqrt()).powi(3);
        assert!(chi < limit, "chi2 = {chi:.1} over limit {limit:.1} with {cells} cells");
    }

    #[test]
    fn samples_stay_inside_the_tail_bound() {
        assert!(draws(1_000_000, 3).iter().all(|x| x.abs() <= TAIL));
    }

    #[test]
    fn table_is_sorted_and_accounts_for_all_the_mass() {
        assert!(CDT.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(CDT.len(), (2 * TAIL) as usize);
        assert_eq!(CDT[0], 0);
        assert!(CDT[CDT.len() - 1] >= u64::MAX - 1);
    }
}
