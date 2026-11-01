//! Raw throughput of the primitives on this machine.
use std::time::Instant;
use unsaid_lwe::{LWE_DIM, Matrix, Prepared, PublicMatrix};

fn main() {
    // A database about the size of a 250k document ranking matrix.
    let (rows, cols) = (2048usize, 19200usize);
    let data: Vec<i8> = (0..rows * cols).map(|i| (i.wrapping_mul(2654435761) >> 7) as i8).collect();
    let m = Matrix::new(rows, cols, data);
    let a = PublicMatrix::new([1u8; 32], cols);

    let t = Instant::now();
    let hint = m.hint(&a);
    println!("hint        {:>8.1} ms   ({rows} x {cols} x {LWE_DIM})", t.elapsed().as_secs_f64() * 1e3);

    let t = Instant::now();
    let token = Prepared::new(&a).unwrap();
    let dt = t.elapsed().as_secs_f64();
    println!("prepare     {:>8.1} ms   ({:.1} GB of A per second)", dt * 1e3, (cols * LWE_DIM * 4) as f64 / dt / 1e9);

    let t = Instant::now();
    let (secret, ct) = token.seal(11, &[(0, 1), (1, 2)]);
    println!("seal        {:>8.3} ms", t.elapsed().as_secs_f64() * 1e3);

    let t = Instant::now();
    let mut ans = Vec::new();
    let reps = 20;
    for _ in 0..reps {
        ans = m.apply(&ct);
    }
    let dt = t.elapsed().as_secs_f64() / reps as f64;
    println!("apply       {:>8.2} ms   ({:.1} GB scanned per second, all threads)", dt * 1e3, (rows * cols) as f64 / dt / 1e9);

    let t = Instant::now();
    let out = secret.open_signed(11, &ans, &hint);
    println!("open        {:>8.2} ms   ({} entries, first {})", t.elapsed().as_secs_f64() * 1e3, out.len(), out[0]);
}
