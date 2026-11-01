//! Spherical k-means, capacity-limited assignment and cluster ordering.
//!
//! Every cluster becomes one block of the ranking matrix, and the matrix needs as many rows
//! as its largest cluster has documents. Those rows set the size of the hint and of every
//! answer, so one oversized cluster would make every client pay for it. That is why clusters
//! are balanced rather than left as k-means produces them.

use rayon::prelude::*;
use unsaid_lwe::chacha::Drbg;

use crate::linalg::{dot, normalize};

fn seed_rng(seed: u64, label: u64) -> Drbg {
    let mut bytes = [0u8; 32];
    bytes[..8].copy_from_slice(&seed.to_le_bytes());
    bytes[8..16].copy_from_slice(&label.to_le_bytes());
    Drbg::from_seed(&bytes)
}

/// Choose initial centres by k-means++: each new centre is drawn with probability
/// proportional to how badly the existing centres explain a point. Uniform random seeding
/// can put two centres in one natural group and leave another without any, and Lloyd
/// iterations rarely recover from that.
fn seed_centres(data: &[f32], dim: usize, k: usize, rng: &mut Drbg) -> Vec<f32> {
    let n = data.len() / dim;
    let mut centres = Vec::with_capacity(k * dim);
    let first = (rng.next_u64() % n as u64) as usize;
    centres.extend_from_slice(&data[first * dim..(first + 1) * dim]);
    // 1 - cos is proportional to the squared distance between unit vectors.
    let mut distance: Vec<f64> = data
        .par_chunks(dim)
        .map(|p| f64::from(1.0 - dot(p, &centres[..dim])).max(0.0))
        .collect();
    while centres.len() < k * dim {
        let total: f64 = distance.iter().sum();
        let chosen = if total > 1e-12 {
            let mut target = (rng.next_u64() as f64 / u64::MAX as f64) * total;
            let mut pick = n - 1;
            for (i, &d) in distance.iter().enumerate() {
                target -= d;
                if target <= 0.0 && d > 0.0 {
                    pick = i;
                    break;
                }
            }
            pick
        } else {
            // Every point coincides with a centre, so any point is as good as another.
            (rng.next_u64() % n as u64) as usize
        };
        let start = centres.len();
        centres.extend_from_slice(&data[chosen * dim..(chosen + 1) * dim]);
        let centre = &centres[start..];
        distance
            .par_iter_mut()
            .zip(data.par_chunks(dim))
            .for_each(|(d, p)| {
                *d = d.min(f64::from(1.0 - dot(p, centre)).max(0.0));
            });
    }
    centres
}

/// Index and score of the most similar centre.
#[inline]
fn nearest(point: &[f32], centres: &[f32], dim: usize) -> (usize, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (c, centre) in centres.chunks_exact(dim).enumerate() {
        let s = dot(point, centre);
        if s > best.1 {
            best = (c, s);
        }
    }
    best
}

/// One chunk's running totals in a k-means pass: centre sums, counts, and assignments.
type ChunkTotals = (Vec<f64>, Vec<u32>, Vec<(f32, u32)>);

/// Spherical k-means on unit vectors. Returns `k` unit centres, row-major.
///
/// # Panics
/// If `k` is zero or there are fewer points than `k`.
pub fn kmeans(data: &[f32], dim: usize, k: usize, iterations: usize, seed: u64) -> Vec<f32> {
    let n = data.len() / dim;
    assert!(k >= 1 && n >= k, "need at least k points");
    let mut rng = seed_rng(seed, k as u64);
    let mut centres = seed_centres(data, dim, k, &mut rng);
    for _ in 0..iterations {
        // Fixed-size chunks summed in order keep this independent of thread count.
        let parts: Vec<ChunkTotals> = data
            .par_chunks(dim * 1024)
            .enumerate()
            .map(|(chunk_index, chunk)| {
                let mut sums = vec![0f64; k * dim];
                let mut counts = vec![0u32; k];
                let mut worst = Vec::new();
                for (i, point) in chunk.chunks_exact(dim).enumerate() {
                    let (c, score) = nearest(point, &centres, dim);
                    counts[c] += 1;
                    for (s, &v) in sums[c * dim..(c + 1) * dim].iter_mut().zip(point) {
                        *s += f64::from(v);
                    }
                    worst.push((score, (chunk_index * 1024 + i) as u32));
                }
                worst.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                worst.truncate(k);
                (sums, counts, worst)
            })
            .collect();
        let mut sums = vec![0f64; k * dim];
        let mut counts = vec![0u32; k];
        let mut worst: Vec<(f32, u32)> = Vec::new();
        for (s, c, w) in parts {
            sums.iter_mut().zip(&s).for_each(|(a, b)| *a += b);
            counts.iter_mut().zip(&c).for_each(|(a, b)| *a += b);
            worst.extend(w);
        }
        worst.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut spare = worst.into_iter();
        for c in 0..k {
            let target = &mut centres[c * dim..(c + 1) * dim];
            if counts[c] == 0 {
                // An empty cluster restarts from a point its own centre explains worst.
                if let Some((_, i)) = spare.next() {
                    let i = i as usize;
                    target.copy_from_slice(&data[i * dim..(i + 1) * dim]);
                }
                continue;
            }
            for (t, &s) in target.iter_mut().zip(&sums[c * dim..(c + 1) * dim]) {
                *t = s as f32;
            }
            normalize(target);
        }
    }
    centres
}

/// Preferences kept per point during assignment.
const PREFERENCES: usize = 16;

/// Assign points to centres so that no centre takes more than `capacity`.
///
/// Points that care most, meaning those whose best centre is furthest ahead of their second
/// best, choose first. Returns the centre of every point.
///
/// # Panics
/// If `capacity * centres` is smaller than the number of points.
pub fn balanced_assign(data: &[f32], dim: usize, centres: &[f32], capacity: usize) -> Vec<u32> {
    let n = data.len() / dim;
    let k = centres.len() / dim;
    assert!(capacity * k >= n, "not enough room for every point");
    let keep = PREFERENCES.min(k);
    let prefs: Vec<(f32, Vec<u32>)> = data
        .par_chunks(dim)
        .map(|point| {
            let scores: Vec<f32> = centres.chunks_exact(dim).map(|c| dot(point, c)).collect();
            let mut order: Vec<u32> = (0..k as u32).collect();
            let by_score = |a: &u32, b: &u32| {
                scores[*b as usize]
                    .total_cmp(&scores[*a as usize])
                    .then(a.cmp(b))
            };
            if keep < k {
                order.select_nth_unstable_by(keep - 1, by_score);
                order.truncate(keep);
            }
            order.sort_by(by_score);
            let margin = if keep > 1 {
                scores[order[0] as usize] - scores[order[1] as usize]
            } else {
                0.0
            };
            (margin, order)
        })
        .collect();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by(|&a, &b| {
        prefs[b as usize]
            .0
            .total_cmp(&prefs[a as usize].0)
            .then(a.cmp(&b))
    });
    let mut counts = vec![0usize; k];
    let mut assign = vec![0u32; n];
    for i in order {
        let i = i as usize;
        let choice = prefs[i]
            .1
            .iter()
            .copied()
            .find(|&c| counts[c as usize] < capacity);
        let c = choice.unwrap_or_else(|| {
            // Every preferred centre is full, so take the best centre that has room.
            let point = &data[i * dim..(i + 1) * dim];
            centres
                .chunks_exact(dim)
                .enumerate()
                .filter(|(c, _)| counts[*c] < capacity)
                .map(|(c, centre)| (c as u32, dot(point, centre)))
                .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
                .map(|(c, _)| c)
                .expect("capacity was checked above")
        });
        counts[c as usize] += 1;
        assign[i] = c;
    }
    assign
}

/// Order points so that neighbours in the list are similar, by walking to the nearest
/// unvisited point. Similar documents then land on the same page, so one private page fetch
/// tends to return several of the best results.
pub fn neighbour_chain(data: &[f32], dim: usize, members: &[u32], centre: &[f32]) -> Vec<u32> {
    if members.is_empty() {
        return Vec::new();
    }
    let point = |m: u32| &data[m as usize * dim..(m as usize + 1) * dim];
    let mut left: Vec<u32> = members.to_vec();
    let start = (0..left.len())
        .max_by(|&a, &b| {
            dot(point(left[a]), centre)
                .total_cmp(&dot(point(left[b]), centre))
                .then(left[b].cmp(&left[a]))
        })
        .unwrap_or(0);
    let mut current = left.swap_remove(start);
    let mut chain = Vec::with_capacity(members.len());
    chain.push(current);
    while !left.is_empty() {
        let here = point(current);
        let next = (0..left.len())
            .max_by(|&a, &b| {
                dot(here, point(left[a]))
                    .total_cmp(&dot(here, point(left[b])))
                    .then(left[b].cmp(&left[a]))
            })
            .unwrap_or(0);
        current = left.swap_remove(next);
        chain.push(current);
    }
    chain
}

/// Routing prototypes for one cluster: up to `count` unit vectors summarising its members.
pub fn prototypes(data: &[f32], dim: usize, members: &[u32], count: usize, seed: u64) -> Vec<f32> {
    let mut points = Vec::with_capacity(members.len() * dim);
    for &m in members {
        points.extend_from_slice(&data[m as usize * dim..(m as usize + 1) * dim]);
    }
    let k = count.min(members.len() / 4).max(1);
    if members.is_empty() {
        return vec![0.0; dim];
    }
    if k == 1 {
        let mut mean = vec![0f32; dim];
        for p in points.chunks_exact(dim) {
            mean.iter_mut().zip(p).for_each(|(a, b)| *a += b);
        }
        normalize(&mut mean);
        return mean;
    }
    kmeans(&points, dim, k, 8, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit vectors scattered around `groups` well separated directions.
    fn blobs(groups: usize, per: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = seed_rng(seed, 99);
        let mut data = Vec::new();
        for g in 0..groups {
            for _ in 0..per {
                let mut v = vec![0f32; dim];
                v[g % dim] = 4.0;
                for x in v.iter_mut() {
                    *x += (rng.next_u32() as f32 / u32::MAX as f32 - 0.5) * 0.5;
                }
                normalize(&mut v);
                data.extend_from_slice(&v);
            }
        }
        data
    }

    #[test]
    fn kmeans_recovers_separated_groups() {
        let (groups, per, dim) = (5, 40, 8);
        let data = blobs(groups, per, dim, 1);
        let centres = kmeans(&data, dim, groups, 20, 7);
        let labels: Vec<usize> = data
            .chunks_exact(dim)
            .map(|p| nearest(p, &centres, dim).0)
            .collect();
        for g in 0..groups {
            let first = labels[g * per];
            assert!(
                labels[g * per..(g + 1) * per].iter().all(|&l| l == first),
                "group {g} split"
            );
        }
        let mut distinct: Vec<usize> = (0..groups).map(|g| labels[g * per]).collect();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), groups);
    }

    #[test]
    fn centres_are_unit_vectors() {
        let data = blobs(3, 30, 6, 2);
        let centres = kmeans(&data, 6, 3, 10, 3);
        for c in centres.chunks_exact(6) {
            assert!((dot(c, c) - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn kmeans_is_deterministic_and_independent_of_thread_count() {
        let data = blobs(6, 500, 16, 3);
        let a = kmeans(&data, 16, 6, 5, 11);
        let b = kmeans(&data, 16, 6, 5, 11);
        assert_eq!(a, b);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let c = pool.install(|| kmeans(&data, 16, 6, 5, 11));
        assert_eq!(a, c);
    }

    #[test]
    fn identical_points_do_not_break_clustering() {
        let data = [1.0f32, 0.0].repeat(20);
        let centres = kmeans(&data, 2, 4, 5, 1);
        assert!(centres.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn k_equal_to_n_works() {
        let data = blobs(2, 2, 4, 9);
        assert_eq!(kmeans(&data, 4, 4, 3, 1).len(), 16);
    }

    #[test]
    fn assignment_respects_capacity_and_covers_everyone() {
        let data = blobs(4, 100, 8, 4);
        let centres = kmeans(&data, 8, 4, 10, 5);
        let assign = balanced_assign(&data, 8, &centres, 110);
        let mut counts = [0usize; 4];
        for &a in &assign {
            counts[a as usize] += 1;
        }
        assert_eq!(counts.iter().sum::<usize>(), 400);
        assert!(counts.iter().all(|&c| c <= 110), "{counts:?}");
    }

    #[test]
    fn assignment_handles_a_lopsided_dataset_at_exact_capacity() {
        // Nearly everything wants the same centre, and the capacity leaves no slack.
        let mut data = blobs(1, 95, 8, 5);
        data.extend(blobs(2, 5, 8, 6).iter().skip(40).copied());
        let n = data.len() / 8;
        let centres = kmeans(&data, 8, 4, 5, 2);
        let capacity = n.div_ceil(4);
        let assign = balanced_assign(&data, 8, &centres, capacity);
        let mut counts = [0usize; 4];
        for &a in &assign {
            counts[a as usize] += 1;
        }
        assert!(counts.iter().all(|&c| c <= capacity), "{counts:?}");
    }

    #[test]
    fn assignment_keeps_points_with_their_natural_group_when_there_is_room() {
        let (groups, per, dim) = (4, 50, 8);
        let data = blobs(groups, per, dim, 7);
        let centres = kmeans(&data, dim, groups, 15, 8);
        let assign = balanced_assign(&data, dim, &centres, per + 10);
        for g in 0..groups {
            let slice = &assign[g * per..(g + 1) * per];
            assert!(slice.iter().all(|&a| a == slice[0]), "group {g} was split");
        }
    }

    #[test]
    fn neighbour_chain_visits_everyone_once_and_groups_similar_points() {
        let (groups, per, dim) = (3, 20, 6);
        let data = blobs(groups, per, dim, 8);
        let members: Vec<u32> = (0..(groups * per) as u32).collect();
        let centre = vec![1.0 / (dim as f32).sqrt(); dim];
        let chain = neighbour_chain(&data, dim, &members, &centre);
        let mut sorted = chain.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, members);
        // A chain through three tight groups changes group only rarely.
        let changes = chain
            .windows(2)
            .filter(|w| w[0] as usize / per != w[1] as usize / per)
            .count();
        assert!(changes <= 4, "{changes} group changes");
    }

    #[test]
    fn neighbour_chain_of_nothing_is_empty() {
        assert!(neighbour_chain(&[], 4, &[], &[0.0; 4]).is_empty());
    }

    #[test]
    fn prototypes_summarise_a_cluster() {
        let data = blobs(4, 25, 8, 9);
        let members: Vec<u32> = (0..100).collect();
        let protos = prototypes(&data, 8, &members, 4, 1);
        assert_eq!(protos.len(), 4 * 8);
        // Each group is close to some prototype.
        for g in 0..4 {
            let p = &data[g * 25 * 8..(g * 25 + 1) * 8];
            assert!(nearest(p, &protos, 8).1 > 0.9);
        }
    }

    #[test]
    fn tiny_clusters_get_one_prototype() {
        let data = blobs(1, 3, 4, 10);
        assert_eq!(prototypes(&data, 4, &[0, 1, 2], 16, 1).len(), 4);
        assert_eq!(prototypes(&data, 4, &[], 16, 1).len(), 4);
    }
}
