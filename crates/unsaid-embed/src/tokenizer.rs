//! The BERT "uncased" tokenizer: text normalisation followed by greedy WordPiece.

use std::collections::HashMap;

use unicode_general_category::{GeneralCategory as Gc, get_general_category as category};
use unicode_normalization::UnicodeNormalization;

/// Words longer than this become `[UNK]`, as in the reference implementation.
const MAX_CHARS_PER_WORD: usize = 100;

/// Turns text into model input ids.
pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    unk: u32,
    cls: u32,
    sep: u32,
    lowercase: bool,
}

fn is_control(c: char) -> bool {
    if matches!(c, '\t' | '\n' | '\r') {
        return false;
    }
    matches!(
        category(c),
        Gc::Control | Gc::Format | Gc::Surrogate | Gc::PrivateUse | Gc::Unassigned
    )
}

fn is_whitespace(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r') || category(c) == Gc::SpaceSeparator
}

fn is_punctuation(c: char) -> bool {
    let cp = c as u32;
    // ASCII symbols such as $ and ^ count as punctuation here, as they do in BERT.
    if (33..=47).contains(&cp)
        || (58..=64).contains(&cp)
        || (91..=96).contains(&cp)
        || (123..=126).contains(&cp)
    {
        return true;
    }
    matches!(
        category(c),
        Gc::ConnectorPunctuation
            | Gc::DashPunctuation
            | Gc::OpenPunctuation
            | Gc::ClosePunctuation
            | Gc::InitialPunctuation
            | Gc::FinalPunctuation
            | Gc::OtherPunctuation
    )
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F
        | 0x2B740..=0x2B81F | 0x2B820..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}

impl Tokenizer {
    /// Build from `vocab.txt` contents, one token per line with the id equal to the line.
    pub fn from_vocab<S: AsRef<str>>(tokens: &[S], lowercase: bool) -> Option<Self> {
        let vocab: HashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.as_ref().to_owned(), i as u32))
            .collect();
        Some(Self {
            unk: *vocab.get("[UNK]")?,
            cls: *vocab.get("[CLS]")?,
            sep: *vocab.get("[SEP]")?,
            vocab,
            lowercase,
        })
    }

    /// Split text into basic tokens before WordPiece.
    fn basic(&self, text: &str) -> Vec<String> {
        let mut cleaned = String::with_capacity(text.len() + 8);
        for c in text.chars() {
            if c == '\0' || c == '\u{fffd}' || is_control(c) {
                continue;
            }
            if is_whitespace(c) {
                cleaned.push(' ');
            } else if is_cjk(c) {
                cleaned.push(' ');
                cleaned.push(c);
                cleaned.push(' ');
            } else {
                cleaned.push(c);
            }
        }
        let mut out = Vec::new();
        for word in cleaned.split_whitespace() {
            let normal: String = if self.lowercase {
                word.to_lowercase()
                    .nfd()
                    .filter(|&c| category(c) != Gc::NonspacingMark)
                    .collect()
            } else {
                word.to_owned()
            };
            let mut current = String::new();
            for c in normal.chars() {
                if is_punctuation(c) {
                    if !current.is_empty() {
                        out.push(std::mem::take(&mut current));
                    }
                    out.push(c.to_string());
                } else {
                    current.push(c);
                }
            }
            if !current.is_empty() {
                out.push(current);
            }
        }
        out
    }

    fn wordpiece(&self, word: &str, out: &mut Vec<u32>) {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() > MAX_CHARS_PER_WORD {
            out.push(self.unk);
            return;
        }
        let start_len = out.len();
        let mut start = 0;
        while start < chars.len() {
            let mut end = chars.len();
            let mut found = None;
            while start < end {
                let mut piece: String = chars[start..end].iter().collect();
                if start > 0 {
                    piece.insert_str(0, "##");
                }
                if let Some(&id) = self.vocab.get(&piece) {
                    found = Some(id);
                    break;
                }
                end -= 1;
            }
            match found {
                Some(id) => out.push(id),
                None => {
                    out.truncate(start_len);
                    out.push(self.unk);
                    return;
                }
            }
            start = end;
        }
    }

    /// Model input ids, `[CLS] ... [SEP]`, truncated so the total is at most `max_len`.
    pub fn encode(&self, text: &str, max_len: usize) -> Vec<u32> {
        let room = max_len.saturating_sub(2);
        let mut ids = Vec::with_capacity(64);
        ids.push(self.cls);
        for word in self.basic(text) {
            self.wordpiece(&word, &mut ids);
            if ids.len() > room {
                break;
            }
        }
        ids.truncate(room + 1);
        ids.push(self.sep);
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok() -> Tokenizer {
        let vocab = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "the", "cat", "sat", "un", "##want", "##ed", ",",
            "run", "##ning", "!", "cafe", "'", "s", "中", "文",
        ];
        Tokenizer::from_vocab(&vocab, true).unwrap()
    }

    fn pieces(t: &Tokenizer, text: &str) -> Vec<u32> {
        let ids = t.encode(text, 100);
        ids[1..ids.len() - 1].to_vec()
    }

    #[test]
    fn wraps_in_special_tokens_and_lowercases() {
        let t = tok();
        assert_eq!(t.encode("The CAT", 16), vec![2, 4, 5, 3]);
    }

    #[test]
    fn splits_words_into_longest_matching_pieces() {
        let t = tok();
        assert_eq!(pieces(&t, "unwanted running"), vec![7, 8, 9, 11, 12]);
    }

    #[test]
    fn unknown_words_become_one_unk_even_if_a_prefix_matched() {
        let t = tok();
        assert_eq!(pieces(&t, "unwantedly cat"), vec![1, 5]);
    }

    #[test]
    fn punctuation_is_separated() {
        let t = tok();
        assert_eq!(pieces(&t, "cat,sat!"), vec![5, 10, 6, 13]);
    }

    #[test]
    fn accents_are_stripped_when_lowercasing() {
        let t = tok();
        assert_eq!(pieces(&t, "CAFÉ"), vec![14]);
    }

    #[test]
    fn cjk_characters_are_separate_tokens() {
        let t = tok();
        assert_eq!(pieces(&t, "中文"), vec![17, 18]);
    }

    #[test]
    fn control_characters_vanish_and_tabs_are_whitespace() {
        let t = tok();
        assert_eq!(pieces(&t, "the\u{0}\u{7}\tcat\nsat"), vec![4, 5, 6]);
    }

    #[test]
    fn truncation_keeps_the_final_separator() {
        let t = tok();
        let ids = t.encode("the cat sat the cat sat the cat sat", 5);
        assert_eq!(ids, vec![2, 4, 5, 6, 3]);
        assert_eq!(t.encode("", 5), vec![2, 3]);
    }

    #[test]
    fn a_huge_word_is_one_unk_and_does_not_loop() {
        let t = tok();
        assert_eq!(pieces(&t, &"a".repeat(5000)), vec![1]);
    }

    #[test]
    fn missing_special_tokens_are_reported() {
        assert!(Tokenizer::from_vocab(&["a", "b"], true).is_none());
    }
}
