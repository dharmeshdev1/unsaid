//! The packed model format and the public embedding interface.
//!
//! A model file holds the architecture, the vocabulary and the weights. Weight matrices can
//! be stored as integers with one scale per row, which shrinks the file a good deal while
//! keeping the embeddings close to the original. Weights are expanded back to floats when
//! loaded, so inference code never sees the difference.

use serde::{Deserialize, Serialize};

use crate::bert::{Bert, Layer, LayerNorm, Linear, Pooling};
use crate::tokenizer::Tokenizer;

const MAGIC: &[u8; 4] = b"USM2";

/// How weight matrices are stored in a packed model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WeightFormat {
    /// 32-bit floats, the same values as the source model.
    F32,
    /// 16-bit integers with one scale per row.
    Q16,
    /// 8-bit integers with one scale per row.
    Q8,
}

/// Why a model could not be loaded or packed.
#[derive(Debug)]
pub struct ModelError(pub String);

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid model: {}", self.0)
    }
}

impl std::error::Error for ModelError {}

fn err(message: impl Into<String>) -> ModelError {
    ModelError(message.into())
}

#[derive(Serialize, Deserialize)]
struct TensorInfo {
    name: String,
    shape: Vec<usize>,
    format: WeightFormat,
}

impl TensorInfo {
    /// Number of rows that have their own scale.
    fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(1)
    }

    fn count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Bytes this tensor occupies in the file.
    fn bytes(&self) -> usize {
        match self.format {
            WeightFormat::F32 => self.count() * 4,
            WeightFormat::Q16 => self.rows() * 4 + self.count() * 2,
            WeightFormat::Q8 => self.rows() * 4 + self.count(),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Header {
    name: String,
    hidden: usize,
    layers: usize,
    heads: usize,
    pooling: Pooling,
    lowercase: bool,
    query_prefix: String,
    document_prefix: String,
    max_tokens: usize,
    vocab: Vec<String>,
    tensors: Vec<TensorInfo>,
}

/// Settings recorded in a packed model.
pub struct PackOptions {
    /// A short identifier, for example `bge-small-en-v1.5`.
    pub name: String,
    /// How token vectors become a sentence vector.
    pub pooling: Pooling,
    /// Text put in front of every query.
    pub query_prefix: String,
    /// Text put in front of every document.
    pub document_prefix: String,
    /// Longest input the model accepts when encoding documents.
    pub max_tokens: usize,
    /// Storage for weight matrices. One-dimensional tensors are always stored as floats.
    pub weights: WeightFormat,
}

/// Convert a Hugging Face `model.safetensors` plus `config.json` and `vocab.txt` into a
/// packed model file.
pub fn pack_safetensors(
    safetensors: &[u8],
    config_json: &str,
    vocab_txt: &str,
    options: &PackOptions,
) -> Result<Vec<u8>, ModelError> {
    let config: serde_json::Value =
        serde_json::from_str(config_json).map_err(|e| err(format!("config.json: {e}")))?;
    let number = |key: &str| {
        config[key]
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| err(format!("config lacks {key}")))
    };
    let (hidden, layers, heads) = (
        number("hidden_size")?,
        number("num_hidden_layers")?,
        number("num_attention_heads")?,
    );
    let lowercase = config["do_lower_case"].as_bool().unwrap_or(true);

    if safetensors.len() < 8 {
        return Err(err("safetensors file is truncated"));
    }
    let header_len = u64::from_le_bytes(safetensors[..8].try_into().unwrap()) as usize;
    let body_start = header_len
        .checked_add(8)
        .filter(|&e| e <= safetensors.len());
    let body_start = body_start.ok_or_else(|| err("safetensors header is truncated"))?;
    let meta: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&safetensors[8..body_start])
            .map_err(|e| err(format!("safetensors header: {e}")))?;
    let body = &safetensors[body_start..];

    let mut names: Vec<&String> = meta.keys().filter(|k| *k != "__metadata__").collect();
    names.sort();
    let mut tensors = Vec::new();
    let mut data = Vec::new();
    for key in names {
        let name = key.strip_prefix("bert.").unwrap_or(key).to_owned();
        let info = &meta[key];
        if name.starts_with("pooler.") || info["dtype"] != "F32" {
            continue;
        }
        let shape: Vec<usize> = info["shape"]
            .as_array()
            .ok_or_else(|| err("tensor without a shape"))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let range = info["data_offsets"]
            .as_array()
            .ok_or_else(|| err("tensor without offsets"))?;
        let (a, b) = (
            range[0].as_u64().unwrap_or(0) as usize,
            range[1].as_u64().unwrap_or(0) as usize,
        );
        let raw = body
            .get(a..b)
            .ok_or_else(|| err("tensor offsets are out of range"))?;
        let values: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // One-dimensional tensors (biases and norm parameters) are few and sensitive.
        let format = if shape.len() >= 2 {
            options.weights
        } else {
            WeightFormat::F32
        };
        match format {
            WeightFormat::F32 => data.extend(values.iter().flat_map(|v| v.to_le_bytes())),
            WeightFormat::Q16 | WeightFormat::Q8 => {
                let row_len = values.len() / shape[0].max(1);
                let top = if format == WeightFormat::Q16 {
                    32767.0
                } else {
                    127.0
                };
                let scales: Vec<f32> = values
                    .chunks_exact(row_len.max(1))
                    .map(|row| row.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12) / top)
                    .collect();
                data.extend(scales.iter().flat_map(|v| v.to_le_bytes()));
                for (row, scale) in values.chunks_exact(row_len.max(1)).zip(&scales) {
                    for v in row {
                        let q = (v / scale).round();
                        if format == WeightFormat::Q16 {
                            data.extend((q as i16).to_le_bytes());
                        } else {
                            data.push(q as i8 as u8);
                        }
                    }
                }
            }
        }
        tensors.push(TensorInfo {
            name,
            shape,
            format,
        });
    }

    let header = Header {
        name: options.name.clone(),
        hidden,
        layers,
        heads,
        pooling: options.pooling,
        lowercase,
        query_prefix: options.query_prefix.clone(),
        document_prefix: options.document_prefix.clone(),
        max_tokens: options.max_tokens,
        vocab: vocab_txt.lines().map(str::to_owned).collect(),
        tensors,
    };
    let header = serde_json::to_vec(&header).map_err(|e| err(e.to_string()))?;
    let mut out = Vec::with_capacity(8 + header.len() + data.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(header.len() as u32).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(&data);
    Ok(out)
}

/// Loaded tensors, looked up by name while the network is assembled.
struct Tensors<'a> {
    infos: &'a [TensorInfo],
    data: &'a [u8],
}

impl Tensors<'_> {
    fn get(&self, name: &str) -> Result<(Vec<f32>, &[usize]), ModelError> {
        let mut offset = 0usize;
        for info in self.infos {
            let bytes = info.bytes();
            if info.name == name {
                let raw = self
                    .data
                    .get(offset..offset + bytes)
                    .ok_or_else(|| err("model file is truncated"))?;
                return Ok((decode(info, raw), &info.shape));
            }
            offset += bytes;
        }
        Err(err(format!("missing tensor {name}")))
    }

    fn linear(&self, prefix: &str) -> Result<Linear, ModelError> {
        let (w, shape) = self.get(&format!("{prefix}.weight"))?;
        let (bias, _) = self.get(&format!("{prefix}.bias"))?;
        let [outputs, inputs] = shape else {
            return Err(err(format!("{prefix} is not a matrix")));
        };
        let (outputs, inputs) = (*outputs, *inputs);
        let mut weight = vec![0f32; w.len()];
        for o in 0..outputs {
            for i in 0..inputs {
                weight[i * outputs + o] = w[o * inputs + i];
            }
        }
        Ok(Linear {
            weight,
            bias,
            inputs,
            outputs,
        })
    }

    fn norm(&self, prefix: &str) -> Result<LayerNorm, ModelError> {
        Ok(LayerNorm {
            gamma: self.get(&format!("{prefix}.weight"))?.0,
            beta: self.get(&format!("{prefix}.bias"))?.0,
        })
    }
}

fn floats(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn decode(info: &TensorInfo, raw: &[u8]) -> Vec<f32> {
    if info.format == WeightFormat::F32 {
        return floats(raw);
    }
    let rows = info.rows();
    let row_len = info.count() / rows.max(1);
    let (scales, body) = raw.split_at(rows * 4);
    let scales = floats(scales);
    let mut out = Vec::with_capacity(info.count());
    match info.format {
        WeightFormat::Q16 => {
            for (row, scale) in body.chunks_exact(row_len * 2).zip(&scales) {
                out.extend(
                    row.chunks_exact(2)
                        .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) * scale),
                );
            }
        }
        _ => {
            for (row, scale) in body.chunks_exact(row_len).zip(&scales) {
                out.extend(row.iter().map(|&b| f32::from(b as i8) * scale));
            }
        }
    }
    out
}

/// A sentence encoder ready to use. It is immutable and safe to share across threads.
pub struct Embedder {
    name: String,
    bert: Bert,
    tokenizer: Tokenizer,
    query_prefix: String,
    document_prefix: String,
    max_tokens: usize,
}

impl Embedder {
    /// Load a packed model.
    pub fn load(file: &[u8]) -> Result<Self, ModelError> {
        if file.len() < 8 || &file[..4] != MAGIC {
            return Err(err("not a packed model file"));
        }
        let header_len = u32::from_le_bytes(file[4..8].try_into().unwrap()) as usize;
        let header_end = 8usize.checked_add(header_len).filter(|&e| e <= file.len());
        let header_end = header_end.ok_or_else(|| err("model header is truncated"))?;
        let header: Header = serde_json::from_slice(&file[8..header_end])
            .map_err(|e| err(format!("model header: {e}")))?;
        let t = Tensors {
            infos: &header.tensors,
            data: &file[header_end..],
        };

        let tokenizer = Tokenizer::from_vocab(&header.vocab, header.lowercase)
            .ok_or_else(|| err("vocabulary lacks the special tokens"))?;
        if header.heads == 0 || header.hidden % header.heads != 0 {
            return Err(err("hidden size is not divisible by the head count"));
        }
        let mut layers = Vec::with_capacity(header.layers);
        for i in 0..header.layers {
            let p = format!("encoder.layer.{i}");
            layers.push(Layer {
                query: t.linear(&format!("{p}.attention.self.query"))?,
                key: t.linear(&format!("{p}.attention.self.key"))?,
                value: t.linear(&format!("{p}.attention.self.value"))?,
                attn_out: t.linear(&format!("{p}.attention.output.dense"))?,
                attn_norm: t.norm(&format!("{p}.attention.output.LayerNorm"))?,
                ffn_in: t.linear(&format!("{p}.intermediate.dense"))?,
                ffn_out: t.linear(&format!("{p}.output.dense"))?,
                ffn_norm: t.norm(&format!("{p}.output.LayerNorm"))?,
            });
        }
        let (token_types, _) = t.get("embeddings.token_type_embeddings.weight")?;
        let (word, word_shape) = t.get("embeddings.word_embeddings.weight")?;
        let word_rows = word_shape.first().copied().unwrap_or(0);
        let (position, position_shape) = t.get("embeddings.position_embeddings.weight")?;
        let position_rows = position_shape.first().copied().unwrap_or(0);
        // A hostile model file must fail here and not panic later while encoding.
        let h = header.hidden;
        let shapes_ok = word.len() == word_rows * h
            && position.len() == position_rows * h
            && word_rows >= header.vocab.len()
            && position_rows >= header.max_tokens
            && header.max_tokens >= 2
            && token_types.len() >= h
            && layers.iter().all(|l| {
                let square = |x: &Linear| x.inputs == h && x.outputs == h;
                let fits = |x: &Linear| {
                    x.weight.len() == x.inputs * x.outputs && x.bias.len() == x.outputs
                };
                let norm = |n: &LayerNorm| n.gamma.len() == h && n.beta.len() == h;
                square(&l.query)
                    && square(&l.key)
                    && square(&l.value)
                    && square(&l.attn_out)
                    && l.ffn_in.inputs == h
                    && l.ffn_out.inputs == l.ffn_in.outputs
                    && l.ffn_out.outputs == h
                    && fits(&l.query)
                    && fits(&l.key)
                    && fits(&l.value)
                    && fits(&l.attn_out)
                    && fits(&l.ffn_in)
                    && fits(&l.ffn_out)
                    && norm(&l.attn_norm)
                    && norm(&l.ffn_norm)
            });
        if !shapes_ok {
            return Err(err("tensor shapes do not fit the architecture"));
        }
        let bert = Bert {
            hidden: header.hidden,
            heads: header.heads,
            word,
            position,
            token_type: token_types[..header.hidden].to_vec(),
            embed_norm: t.norm("embeddings.LayerNorm")?,
            layers,
            pooling: header.pooling,
        };
        Ok(Self {
            name: header.name,
            bert,
            tokenizer,
            query_prefix: header.query_prefix,
            document_prefix: header.document_prefix,
            max_tokens: header.max_tokens,
        })
    }

    /// The model's identifier.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Length of the vectors this model produces.
    pub fn dim(&self) -> usize {
        self.bert.hidden()
    }

    /// Embed a search query.
    pub fn embed_query(&self, text: &str) -> Vec<f32> {
        let ids = self
            .tokenizer
            .encode(&format!("{}{text}", self.query_prefix), self.max_tokens);
        self.bert.encode(&ids)
    }

    /// Embed a document, keeping at most `max_tokens` tokens of it.
    pub fn embed_document(&self, text: &str, max_tokens: usize) -> Vec<f32> {
        let limit = max_tokens.min(self.max_tokens);
        let ids = self
            .tokenizer
            .encode(&format!("{}{text}", self.document_prefix), limit);
        self.bert.encode(&ids)
    }

    /// Token ids for `text`, without any prefix. Useful for testing the tokenizer.
    pub fn tokenize(&self, text: &str, max_tokens: usize) -> Vec<u32> {
        self.tokenizer.encode(text, max_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn garbage_is_refused_without_panicking() {
        for bytes in [
            &b""[..],
            &b"USM2"[..],
            &b"NOPE12345678"[..],
            &b"USM2\xff\xff\xff\xffab"[..],
        ] {
            assert!(Embedder::load(bytes).is_err());
        }
    }

    #[test]
    fn packing_rejects_truncated_safetensors() {
        let options = PackOptions {
            name: "x".into(),
            pooling: Pooling::Cls,
            query_prefix: String::new(),
            document_prefix: String::new(),
            max_tokens: 16,
            weights: WeightFormat::Q8,
        };
        let config = r#"{"hidden_size":4,"num_hidden_layers":1,"num_attention_heads":2}"#;
        assert!(pack_safetensors(b"abc", config, "", &options).is_err());
        assert!(pack_safetensors(&u64::MAX.to_le_bytes(), config, "", &options).is_err());
        assert!(pack_safetensors(b"", "not json", "", &options).is_err());
    }
}
