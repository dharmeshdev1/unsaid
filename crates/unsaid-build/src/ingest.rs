//! Read documents from disk and split long ones into passages.

use std::fs;
use std::path::{Path, PathBuf};

use unsaid_core::Document;

/// How source files become documents.
#[derive(Clone, Debug)]
pub struct IngestOptions {
    /// Longest passage, in words.
    pub chunk_words: usize,
    /// Words shared by consecutive passages of one file.
    pub overlap_words: usize,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            chunk_words: 120,
            overlap_words: 20,
        }
    }
}

#[derive(serde::Deserialize)]
struct JsonDoc {
    #[serde(default)]
    title: String,
    text: String,
    #[serde(default)]
    url: String,
}

const TEXT_EXTENSIONS: [&str; 8] = ["txt", "md", "markdown", "rst", "text", "html", "htm", "org"];

/// Read documents from a `.jsonl` file (one `{"title","text","url"}` object per line), a
/// single text file, or a directory walked recursively in sorted order.
pub fn read_documents(path: &Path, options: &IngestOptions) -> Result<Vec<Document>, String> {
    let meta = fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut docs = Vec::new();
    if meta.is_dir() {
        let mut files = Vec::new();
        walk(path, &mut files).map_err(|e| format!("{}: {e}", path.display()))?;
        files.sort();
        for file in files {
            read_file(&file, path, options, &mut docs)?;
        }
    } else {
        read_file(path, path.parent().unwrap_or(path), options, &mut docs)?;
    }
    if docs.is_empty() {
        return Err(format!("no documents found in {}", path.display()));
    }
    Ok(docs)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        if name.to_string_lossy().starts_with('.') {
            continue;
        }
        if entry.file_type()?.is_dir() {
            walk(&path, out)?;
        } else {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            if ext == "jsonl" || TEXT_EXTENSIONS.contains(&ext.as_str()) {
                out.push(path);
            }
        }
    }
    Ok(())
}

fn read_file(
    path: &Path,
    root: &Path,
    options: &IngestOptions,
    docs: &mut Vec<Document>,
) -> Result<(), String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let contents = String::from_utf8_lossy(&bytes);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    if ext == "jsonl" {
        for (i, line) in contents.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let row: JsonDoc = serde_json::from_str(line)
                .map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?;
            push_chunks(&row.title, &row.text, &row.url, options, docs);
        }
        return Ok(());
    }
    let relative = path.strip_prefix(root).unwrap_or(path);
    let url = relative.to_string_lossy().replace('\\', "/");
    let (title, text) = if ext == "html" || ext == "htm" {
        let (title, text) = strip_html(&contents);
        (title.unwrap_or_else(|| file_title(path)), text)
    } else {
        (
            first_heading(&contents).unwrap_or_else(|| file_title(path)),
            contents.into_owned(),
        )
    };
    push_chunks(&title, &text, &url, options, docs);
    Ok(())
}

fn file_title(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().replace(['_', '-'], " "))
        .unwrap_or_default()
}

fn first_heading(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .find_map(|l| l.strip_prefix("# ").map(|h| h.trim().to_owned()))
        .filter(|h| !h.is_empty())
}

fn push_chunks(
    title: &str,
    text: &str,
    url: &str,
    options: &IngestOptions,
    docs: &mut Vec<Document>,
) {
    let chunks = chunk_words(text, options.chunk_words, options.overlap_words);
    for chunk in chunks {
        docs.push(Document {
            title: title.to_owned(),
            text: chunk,
            url: url.to_owned(),
        });
    }
}

/// Split text into passages of at most `size` words, sharing `overlap` words between
/// neighbours. Whitespace inside a passage is normalised to single spaces.
pub fn chunk_words(text: &str, size: usize, overlap: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }
    let size = size.max(1);
    let step = size.saturating_sub(overlap).max(1);
    let mut out = Vec::new();
    let mut start = 0;
    loop {
        let end = (start + size).min(words.len());
        out.push(words[start..end].join(" "));
        if end == words.len() {
            return out;
        }
        start += step;
    }
}

/// Remove tags, scripts and styles from HTML, and decode the common entities. Returns the
/// page title, if it has one, and the visible text.
pub fn strip_html(html: &str) -> (Option<String>, String) {
    let mut text = String::with_capacity(html.len() / 2);
    let mut title = None;
    let mut rest = html;
    while let Some(open) = rest.find('<') {
        text.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let name: String = after
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '/' || *c == '!')
            .collect::<String>()
            .to_lowercase();
        if name == "script" || name == "style" || name == "title" {
            let closing = format!("</{name}");
            let lower = after.to_lowercase();
            match lower.find(&closing) {
                Some(end) => {
                    if name == "title" {
                        if let Some(gt) = after[..end].find('>') {
                            let t = decode_entities(after[gt + 1..end].trim());
                            if !t.is_empty() {
                                title = Some(t);
                            }
                        }
                    }
                    let tail = &after[end..];
                    rest = tail.find('>').map_or("", |gt| &tail[gt + 1..]);
                }
                None => rest = "",
            }
            text.push(' ');
            continue;
        }
        match after.find('>') {
            Some(gt) => {
                text.push(' ');
                rest = &after[gt + 1..];
            }
            None => {
                rest = "";
            }
        }
    }
    text.push_str(rest);
    (title, decode_entities(&text))
}

fn decode_entities(text: &str) -> String {
    text.replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_one_passage_with_clean_whitespace() {
        assert_eq!(chunk_words("  a   b\n\nc ", 10, 2), vec!["a b c"]);
    }

    #[test]
    fn long_text_overlaps_and_covers_everything() {
        let text: String = (0..25).map(|i| format!("w{i} ")).collect();
        let chunks = chunk_words(&text, 10, 3);
        assert_eq!(chunks[0].split(' ').count(), 10);
        assert!(chunks[0].ends_with("w9") && chunks[1].starts_with("w7"));
        assert!(chunks.last().unwrap().ends_with("w24"));
    }

    #[test]
    fn empty_text_gives_nothing_and_overlap_cannot_stall() {
        assert!(chunk_words("   ", 10, 2).is_empty());
        assert_eq!(chunk_words("a b c", 2, 5).len(), 2);
        assert_eq!(chunk_words("a b c", 0, 0).len(), 3);
    }

    #[test]
    fn html_loses_tags_scripts_and_styles_and_keeps_the_title() {
        let html = "<html><head><title>My &amp; Page</title><style>p{color:red}</style></head>\
                    <body><h1>Hello</h1><script>var x = '<b>nope</b>';</script><p>World &lt;3</p></body></html>";
        let (title, text) = strip_html(html);
        assert_eq!(title.as_deref(), Some("My & Page"));
        let words: Vec<&str> = text.split_whitespace().collect();
        assert_eq!(words, ["Hello", "World", "<3"]);
    }

    #[test]
    fn malformed_html_does_not_panic_or_lose_text_before_it() {
        for bad in [
            "keep <",
            "keep <b",
            "keep <script>x",
            "<title>x",
            "<>",
            "a < b > c",
            "<é>ü",
        ] {
            let (_, text) = strip_html(bad);
            if bad.starts_with("keep") {
                assert!(text.contains("keep"), "{bad}");
            }
        }
    }

    #[test]
    fn a_directory_is_read_recursively_in_order_skipping_hidden_files_and_other_types() {
        let dir = std::env::temp_dir().join(format!("unsaid-ingest-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("sub")).unwrap();
        fs::create_dir_all(dir.join(".git")).unwrap();
        fs::write(dir.join("b.md"), "# Bee Title\nbuzz").unwrap();
        fs::write(dir.join("a_file.txt"), "ant words").unwrap();
        fs::write(dir.join("sub/c.html"), "<title>Sea</title><p>waves</p>").unwrap();
        fs::write(dir.join("skip.bin"), "x").unwrap();
        fs::write(dir.join(".git/config"), "x").unwrap();
        let docs = read_documents(&dir, &IngestOptions::default()).unwrap();
        let titles: Vec<&str> = docs.iter().map(|d| d.title.as_str()).collect();
        assert_eq!(titles, ["a file", "Bee Title", "Sea"]);
        assert_eq!(docs[2].url, "sub/c.html");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn jsonl_is_parsed_and_errors_name_the_line() {
        let dir = std::env::temp_dir().join(format!("unsaid-jsonl-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let good = dir.join("good.jsonl");
        fs::write(
            &good,
            "{\"title\":\"T\",\"text\":\"body\",\"url\":\"u\"}\n\n{\"text\":\"only\"}\n",
        )
        .unwrap();
        let docs = read_documents(&good, &IngestOptions::default()).unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(
            docs[0],
            Document {
                title: "T".into(),
                text: "body".into(),
                url: "u".into()
            }
        );
        let bad = dir.join("bad.jsonl");
        fs::write(&bad, "{\"text\":\"ok\"}\nnot json\n").unwrap();
        let e = read_documents(&bad, &IngestOptions::default()).unwrap_err();
        assert!(e.contains(":2:"), "{e}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_empty_or_missing_source_is_an_error() {
        let dir = std::env::temp_dir().join(format!("unsaid-empty-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        assert!(read_documents(&dir, &IngestOptions::default()).is_err());
        assert!(read_documents(&dir.join("none"), &IngestOptions::default()).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn invalid_utf8_is_tolerated() {
        let dir = std::env::temp_dir().join(format!("unsaid-utf-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("x.txt"), b"good \xff\xfe words").unwrap();
        let docs = read_documents(&dir, &IngestOptions::default()).unwrap();
        assert!(docs[0].text.contains("good"));
        fs::remove_dir_all(&dir).unwrap();
    }
}
