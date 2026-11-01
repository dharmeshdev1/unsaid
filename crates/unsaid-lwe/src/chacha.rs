//! ChaCha, written so the compiler can vectorise it on every target without `unsafe`.
//!
//! Two uses. A public, seekable expansion of a seed into the LWE matrix `A` (12 rounds,
//! because that matrix is not a secret and clients expand hundreds of megabytes of it).
//! And a deterministic generator for secret randomness seeded from the operating system
//! (20 rounds).

use zeroize::Zeroize;

const LANES: usize = 8;
const WORDS_PER_GROUP: usize = 16 * LANES;
const SIGMA_WORDS: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];

type Lane = [u32; LANES];

#[inline(always)]
fn add(a: &mut Lane, b: &Lane) {
    for i in 0..LANES {
        a[i] = a[i].wrapping_add(b[i]);
    }
}

#[inline(always)]
fn xor_rot(d: &mut Lane, a: &Lane, r: u32) {
    for i in 0..LANES {
        d[i] = (d[i] ^ a[i]).rotate_left(r);
    }
}

#[inline(always)]
fn quarter(a: &mut Lane, b: &mut Lane, c: &mut Lane, d: &mut Lane) {
    add(a, b);
    xor_rot(d, a, 16);
    add(c, d);
    xor_rot(b, c, 12);
    add(a, b);
    xor_rot(d, a, 8);
    add(c, d);
    xor_rot(b, c, 7);
}

/// Produce `LANES` consecutive ChaCha blocks, in block order, starting at `counter`.
#[inline]
fn group<const ROUNDS: usize>(
    key: &[u32; 8],
    nonce: &[u32; 3],
    counter: u32,
    out: &mut [u32; WORDS_PER_GROUP],
) {
    let splat = |v: u32| [v; LANES];
    let mut ctr = [0u32; LANES];
    for (i, c) in ctr.iter_mut().enumerate() {
        *c = counter.wrapping_add(i as u32);
    }
    let init: [Lane; 16] = [
        splat(SIGMA_WORDS[0]),
        splat(SIGMA_WORDS[1]),
        splat(SIGMA_WORDS[2]),
        splat(SIGMA_WORDS[3]),
        splat(key[0]),
        splat(key[1]),
        splat(key[2]),
        splat(key[3]),
        splat(key[4]),
        splat(key[5]),
        splat(key[6]),
        splat(key[7]),
        ctr,
        splat(nonce[0]),
        splat(nonce[1]),
        splat(nonce[2]),
    ];
    let [
        mut x0,
        mut x1,
        mut x2,
        mut x3,
        mut x4,
        mut x5,
        mut x6,
        mut x7,
        mut x8,
        mut x9,
        mut x10,
        mut x11,
        mut x12,
        mut x13,
        mut x14,
        mut x15,
    ] = init;
    for _ in 0..ROUNDS / 2 {
        quarter(&mut x0, &mut x4, &mut x8, &mut x12);
        quarter(&mut x1, &mut x5, &mut x9, &mut x13);
        quarter(&mut x2, &mut x6, &mut x10, &mut x14);
        quarter(&mut x3, &mut x7, &mut x11, &mut x15);
        quarter(&mut x0, &mut x5, &mut x10, &mut x15);
        quarter(&mut x1, &mut x6, &mut x11, &mut x12);
        quarter(&mut x2, &mut x7, &mut x8, &mut x13);
        quarter(&mut x3, &mut x4, &mut x9, &mut x14);
    }
    let finals = [
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15,
    ];
    for lane in 0..LANES {
        for word in 0..16 {
            out[lane * 16 + word] = finals[word][lane].wrapping_add(init[word][lane]);
        }
    }
}

fn key_words(seed: &[u8; 32]) -> [u32; 8] {
    let mut key = [0u32; 8];
    for (k, c) in key.iter_mut().zip(seed.chunks_exact(4)) {
        *k = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    key
}

fn fill<const ROUNDS: usize>(key: &[u32; 8], nonce: &[u32; 3], start: u32, out: &mut [u32]) {
    let mut counter = start;
    let mut buf = [0u32; WORDS_PER_GROUP];
    for chunk in out.chunks_mut(WORDS_PER_GROUP) {
        group::<ROUNDS>(key, nonce, counter, &mut buf);
        chunk.copy_from_slice(&buf[..chunk.len()]);
        counter = counter.wrapping_add(LANES as u32);
    }
}

/// Fill `out` with the ChaCha12 keystream for stream number `stream`.
///
/// Words are emitted directly, which is the same stream RFC 8439 defines if each word is
/// serialised little-endian. Each stream is addressable on its own, so any row of `A` can
/// be produced without producing the rows before it.
pub fn expand12(seed: &[u8; 32], stream: u64, out: &mut [u32]) {
    let key = key_words(seed);
    let nonce = [stream as u32, (stream >> 32) as u32, 0];
    fill::<12>(&key, &nonce, 0, out);
}

/// Random generator for secrets: ChaCha20, keyed once from the operating system.
pub struct Drbg {
    key: [u32; 8],
    counter: u32,
    buf: [u32; WORDS_PER_GROUP],
    pos: usize,
}

impl Drbg {
    /// A generator keyed with fresh operating-system entropy.
    pub fn from_os() -> Result<Self, getrandom::Error> {
        let mut seed = [0u8; 32];
        getrandom::getrandom(&mut seed)?;
        let rng = Self::from_seed(&seed);
        seed.zeroize();
        Ok(rng)
    }

    /// A generator keyed with `seed`. Secret randomness must come from [`Drbg::from_os`].
    /// This exists so tests and test vectors are reproducible.
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        Self {
            key: key_words(seed),
            counter: 0,
            buf: [0; WORDS_PER_GROUP],
            pos: WORDS_PER_GROUP,
        }
    }

    /// Next 32 uniformly random bits.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.pos == WORDS_PER_GROUP {
            group::<20>(&self.key, &[0, 0, 0], self.counter, &mut self.buf);
            self.counter = self
                .counter
                .checked_add(LANES as u32)
                .expect("a generator is rekeyed long before 2^38 bytes");
            self.pos = 0;
        }
        let word = self.buf[self.pos];
        self.pos += 1;
        word
    }

    /// Next 64 uniformly random bits.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        (u64::from(self.next_u32()) << 32) | u64::from(self.next_u32())
    }
}

impl Drop for Drbg {
    fn drop(&mut self) {
        self.key.zeroize();
        self.buf.zeroize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha12Rng;
    use rand_core::{RngCore, SeedableRng};

    /// RFC 8439 section 2.3.2.
    #[test]
    fn matches_the_rfc_8439_block_test_vector() {
        let seed: [u8; 32] = core::array::from_fn(|i| i as u8);
        let key = key_words(&seed);
        let nonce = [0x0900_0000, 0x4a00_0000, 0];
        let mut out = [0u32; WORDS_PER_GROUP];
        group::<20>(&key, &nonce, 1, &mut out);
        let want: [u32; 16] = [
            0xe4e7f110, 0x15593bd1, 0x1fdd0f50, 0xc47120a3, 0xc7f4d1c7, 0x0368c033, 0x9aaa2204,
            0x4e6cd4c3, 0x466482d2, 0x09aa9f07, 0x05d7c214, 0xa2028bd9, 0xd19c12b5, 0xb94e16de,
            0xe883d0cb, 0x4e3c50a2,
        ];
        assert_eq!(&out[..16], &want);
    }

    /// Every lane must equal the block a one-at-a-time implementation would produce.
    #[test]
    fn lanes_are_consecutive_blocks() {
        let seed = [7u8; 32];
        let key = key_words(&seed);
        let nonce = [1, 2, 3];
        let mut wide = [0u32; WORDS_PER_GROUP];
        group::<12>(&key, &nonce, 40, &mut wide);
        for lane in 0..LANES {
            let mut single = [0u32; WORDS_PER_GROUP];
            group::<12>(&key, &nonce, 40 + lane as u32, &mut single);
            assert_eq!(&wide[lane * 16..lane * 16 + 16], &single[..16], "lane {lane}");
        }
    }

    /// The 12-round stream agrees with an independent implementation.
    #[test]
    fn expand12_matches_an_independent_chacha12() {
        let seed: [u8; 32] = core::array::from_fn(|i| (i * 3) as u8);
        for stream in [0u64, 1, 77, 0xdead_beef_1234] {
            let mut ours = vec![0u32; 1000];
            expand12(&seed, stream, &mut ours);
            // rand_chacha lays the state out as a 64-bit counter then a 64-bit stream.
            // Our nonce words (stream_lo, stream_hi, 0) occupy state[13..16], so the
            // equivalent is counter_hi = stream_lo and rand_chacha stream = stream_hi.
            let mut rng = ChaCha12Rng::from_seed(seed);
            rng.set_stream(stream >> 32);
            rng.set_word_pos(u128::from(stream as u32) << 36);
            let theirs: Vec<u32> = (0..1000).map(|_| rng.next_u32()).collect();
            assert_eq!(ours, theirs, "stream {stream}");
        }
    }

    #[test]
    fn expansion_is_deterministic_and_streams_are_independent() {
        let seed = [9u8; 32];
        let mut a = vec![0u32; 500];
        let mut b = vec![0u32; 500];
        let mut c = vec![0u32; 500];
        expand12(&seed, 5, &mut a);
        expand12(&seed, 5, &mut b);
        expand12(&seed, 6, &mut c);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn partial_group_lengths_are_a_prefix_of_the_full_stream() {
        let seed = [1u8; 32];
        let mut full = vec![0u32; WORDS_PER_GROUP * 3];
        expand12(&seed, 0, &mut full);
        for len in [1, 15, 16, 17, WORDS_PER_GROUP - 1, WORDS_PER_GROUP + 1, 300] {
            let mut part = vec![0u32; len];
            expand12(&seed, 0, &mut part);
            assert_eq!(part, full[..len], "length {len}");
        }
    }

    #[test]
    fn drbg_is_reproducible_only_from_the_same_seed() {
        let mut a = Drbg::from_seed(&[3u8; 32]);
        let mut b = Drbg::from_seed(&[3u8; 32]);
        let mut c = Drbg::from_seed(&[4u8; 32]);
        for _ in 0..1000 {
            let x = a.next_u64();
            assert_eq!(x, b.next_u64());
            assert_ne!(x, c.next_u64());
        }
    }

    #[test]
    fn drbg_from_os_differs_every_time() {
        let mut a = Drbg::from_os().unwrap();
        let mut b = Drbg::from_os().unwrap();
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn drbg_output_is_balanced() {
        let mut rng = Drbg::from_seed(&[11u8; 32]);
        let mut ones = 0u64;
        let words = 1 << 16;
        for _ in 0..words {
            ones += u64::from(rng.next_u32().count_ones());
        }
        let total = words as f64 * 32.0;
        let z = (ones as f64 - total / 2.0) / (total / 4.0).sqrt();
        assert!(z.abs() < 5.0, "z = {z}");
    }
}
