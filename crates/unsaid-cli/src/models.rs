//! Finding and downloading packed embedding models.

use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use unsaid_embed::KNOWN_MODELS;

const BASE_URL: &str = "https://github.com/dharmeshdev1/unsaid/releases/download/models-v1";

/// Larger than any model on the trusted list. The hash decides what is kept, and this decides how
/// much is read to find out.
const MAX_MODEL_DOWNLOAD: u64 = 1 << 30;

/// Where downloaded files are kept.
pub fn cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("UNSAID_CACHE") {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"));
    home.map_or_else(
        || PathBuf::from(".unsaid-cache"),
        |h| PathBuf::from(h).join(".cache/unsaid"),
    )
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Download `url` to `path`, keeping it only if its hash matches.
fn download(url: &str, path: &Path, sha256: &str) -> Result<(), String> {
    eprintln!("downloading {url}");
    let response = ureq::get(url).call().map_err(|e| format!("{url}: {e}"))?;
    let total: u64 = response
        .header("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut reader = response.into_reader().take(MAX_MODEL_DOWNLOAD);
    let mut data = Vec::with_capacity(total as usize);
    let mut buffer = vec![0u8; 1 << 20];
    let mut last_report = 0u64;
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|e| format!("{url}: {e}"))?;
        if read == 0 {
            break;
        }
        data.extend_from_slice(&buffer[..read]);
        let so_far = data.len() as u64;
        if total > 0 && so_far - last_report > total / 10 {
            eprintln!("  {:>3}%", so_far * 100 / total);
            last_report = so_far;
        }
    }
    let got = hex(&Sha256::digest(&data));
    if got != sha256 {
        return Err(format!("{url} has hash {got}, expected {sha256}"));
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let partial = path.with_extension("partial");
    std::fs::write(&partial, &data).map_err(|e| format!("{}: {e}", partial.display()))?;
    std::fs::rename(&partial, path).map_err(|e| format!("{}: {e}", path.display()))
}

/// Resolve a model given as a short name such as `minilm`, or as a path to a packed file.
pub fn resolve(model: &str) -> Result<Vec<u8>, String> {
    let as_path = Path::new(model);
    if as_path.is_file() {
        return std::fs::read(as_path).map_err(|e| format!("{model}: {e}"));
    }
    let known = KNOWN_MODELS
        .iter()
        .find(|m| m.key == model || m.file.starts_with(model))
        .ok_or_else(|| {
            format!("unknown model {model:?}. Use minilm, bge, or the path to a packed model file")
        })?;
    let path = cache_dir().join("models").join(known.file);
    if !path.is_file() {
        download(&format!("{BASE_URL}/{}", known.file), &path, known.sha256)?;
    }
    let data = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let got = hex(&Sha256::digest(&data));
    if got != known.sha256 {
        return Err(format!(
            "{} has hash {got}, expected {}. Delete it to download it again",
            path.display(),
            known.sha256
        ));
    }
    Ok(data)
}
