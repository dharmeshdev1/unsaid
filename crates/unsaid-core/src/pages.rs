//! Document pages: fixed-size, compressed groups of documents.
//!
//! Private retrieval returns a whole column of the database, so every column must have the
//! same length and the length must not depend on what is inside. A page is that column. Its
//! layout is a two-byte length, a DEFLATE stream of its documents, and zero padding.

use miniz_oxide::deflate::compress_to_vec;
use miniz_oxide::inflate::decompress_to_vec_with_limit;

/// A search result's content.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Document {
    /// Short heading.
    pub title: String,
    /// Passage shown as the result text.
    pub text: String,
    /// Where the document lives, which may be empty.
    #[serde(default)]
    pub url: String,
}

fn put_varint(mut value: usize, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn get_varint(input: &mut &[u8]) -> Option<usize> {
    let mut value = 0usize;
    for shift in (0..35).step_by(7) {
        let (&byte, rest) = input.split_first()?;
        *input = rest;
        value |= usize::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Some(value);
        }
    }
    None
}

fn put_str(s: &str, out: &mut Vec<u8>) {
    put_varint(s.len(), out);
    out.extend_from_slice(s.as_bytes());
}

fn get_str(input: &mut &[u8]) -> Option<String> {
    let len = get_varint(input)?;
    if len > input.len() {
        return None;
    }
    let (head, rest) = input.split_at(len);
    *input = rest;
    String::from_utf8(head.to_vec()).ok()
}

/// Largest size a page may expand to, which bounds what a malicious page can make a client
/// allocate.
const MAX_INFLATED: usize = 1 << 22;

/// Serialise documents without compressing them.
pub fn serialize(documents: &[&Document]) -> Vec<u8> {
    let mut raw = Vec::new();
    for doc in documents {
        put_str(&doc.title, &mut raw);
        put_str(&doc.text, &mut raw);
        put_str(&doc.url, &mut raw);
    }
    raw
}

/// Encode documents into a page of exactly `size` bytes, or `None` if they do not fit.
pub fn encode(documents: &[&Document], size: usize) -> Option<Vec<u8>> {
    let packed = compress_to_vec(&serialize(documents), 7);
    if packed.len() + 2 > size || packed.len() > usize::from(u16::MAX) {
        return None;
    }
    let mut page = Vec::with_capacity(size);
    page.extend_from_slice(&(packed.len() as u16).to_le_bytes());
    page.extend_from_slice(&packed);
    page.resize(size, 0);
    Some(page)
}

/// Decode a page that should hold at most `max_documents` documents. `None` means the page is
/// corrupt, and no input can cause a panic. The limit matters because a small page can expand
/// to over a million empty documents.
pub fn decode(page: &[u8], max_documents: usize) -> Option<Vec<Document>> {
    let (len, body) = page.split_at_checked(2)?;
    let len = usize::from(u16::from_le_bytes([len[0], len[1]]));
    let packed = body.get(..len)?;
    let raw = decompress_to_vec_with_limit(packed, MAX_INFLATED).ok()?;
    let mut input = raw.as_slice();
    let mut documents = Vec::new();
    while !input.is_empty() {
        if documents.len() == max_documents {
            return None;
        }
        let title = get_str(&mut input)?;
        let text = get_str(&mut input)?;
        let url = get_str(&mut input)?;
        documents.push(Document { title, text, url });
    }
    Some(documents)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(i: usize) -> Document {
        Document {
            title: format!("Title {i}"),
            text: format!("Some text about thing {i}, naïve café 日本語 🙂. ").repeat(3),
            url: format!("https://example.org/{i}"),
        }
    }

    #[test]
    fn pages_round_trip_at_exactly_the_requested_size() {
        let docs: Vec<Document> = (0..5).map(doc).collect();
        let refs: Vec<&Document> = docs.iter().collect();
        let page = encode(&refs, 1024).unwrap();
        assert_eq!(page.len(), 1024);
        assert_eq!(decode(&page, usize::MAX).unwrap(), docs);
    }

    #[test]
    fn documents_that_do_not_fit_are_refused() {
        let docs: Vec<Document> = (0..200).map(doc).collect();
        let refs: Vec<&Document> = docs.iter().collect();
        assert!(encode(&refs, 256).is_none());
    }

    #[test]
    fn an_empty_page_is_valid() {
        let page = encode(&[], 64).unwrap();
        assert_eq!(decode(&page, usize::MAX).unwrap(), vec![]);
    }

    #[test]
    fn empty_fields_survive() {
        let d = Document {
            title: String::new(),
            text: String::new(),
            url: String::new(),
        };
        let page = encode(&[&d], 64).unwrap();
        assert_eq!(decode(&page, usize::MAX).unwrap(), vec![d]);
    }

    #[test]
    fn varints_round_trip_across_byte_boundaries() {
        for value in [0usize, 1, 127, 128, 16383, 16384, 1 << 28] {
            let mut out = Vec::new();
            put_varint(value, &mut out);
            let mut input = out.as_slice();
            assert_eq!(get_varint(&mut input), Some(value));
            assert!(input.is_empty());
        }
    }

    #[test]
    fn corrupt_pages_return_none_and_never_panic() {
        let docs: Vec<Document> = (0..3).map(doc).collect();
        let refs: Vec<&Document> = docs.iter().collect();
        let good = encode(&refs, 512).unwrap();
        assert!(decode(&[], usize::MAX).is_none());
        assert!(decode(&[1], usize::MAX).is_none());
        assert!(decode(&[0xff, 0xff, 0, 0], usize::MAX).is_none());
        // Flip bytes throughout: any outcome is acceptable except a panic.
        let mut state = 12345u64;
        for _ in 0..3000 {
            let mut bad = good.clone();
            for _ in 0..4 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let at = (state >> 33) as usize % bad.len();
                bad[at] ^= (state >> 25) as u8 | 1;
            }
            let _ = decode(&bad, usize::MAX);
        }
    }

    #[test]
    fn a_decompression_bomb_is_refused() {
        let huge = vec![0u8; MAX_INFLATED + 1024];
        let packed = compress_to_vec(&huge, 7);
        assert!(packed.len() < u16::MAX as usize);
        let mut page = (packed.len() as u16).to_le_bytes().to_vec();
        page.extend_from_slice(&packed);
        assert!(decode(&page, usize::MAX).is_none());
    }

    /// A long valid document compresses to almost nothing, so only the limit can stop it.
    #[test]
    fn a_bomb_made_of_valid_documents_is_refused() {
        let doc = Document {
            title: "a".repeat(MAX_INFLATED + 1024),
            text: String::new(),
            url: String::new(),
        };
        let packed = compress_to_vec(&serialize(&[&doc]), 7);
        assert!(packed.len() < u16::MAX as usize);
        let mut page = (packed.len() as u16).to_le_bytes().to_vec();
        page.extend_from_slice(&packed);
        assert!(decode(&page, usize::MAX).is_none());
    }

    #[test]
    fn more_documents_than_the_page_should_hold_are_refused() {
        let docs: Vec<Document> = (0..3).map(doc).collect();
        let refs: Vec<&Document> = docs.iter().collect();
        let page = encode(&refs, 4096).unwrap();
        assert!(decode(&page, 3).is_some());
        assert!(decode(&page, 2).is_none());

        // Three zero bytes make an empty document, and they compress extremely well.
        let raw = vec![0u8; 3 * 100_000];
        let packed = compress_to_vec(&raw, 7);
        let mut page = (packed.len() as u16).to_le_bytes().to_vec();
        page.extend_from_slice(&packed);
        assert!(decode(&page, 16).is_none());
    }

    #[test]
    fn oversized_string_lengths_are_refused() {
        let mut raw = Vec::new();
        put_varint(usize::MAX >> 8, &mut raw);
        let packed = compress_to_vec(&raw, 7);
        let mut page = (packed.len() as u16).to_le_bytes().to_vec();
        page.extend_from_slice(&packed);
        assert!(decode(&page, usize::MAX).is_none());
    }
}
