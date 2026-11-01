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

/// Longest query input, in tokens. Questions are short, and this bounds the work one can cause.
const MAX_QUERY_TOKENS: usize = 128;
/// Largest vector length a model may declare.
const MAX_HIDDEN: usize = 1 << 13;

/// A model that has been packed and checked against its reference implementation.
pub struct KnownModel {
    /// Short name for the command line.
    pub key: &'static str,
    /// File name, which is also the model's full name.
    pub file: &'static str,
    /// Hex SHA-256 of the packed file.
    pub sha256: &'static str,
}

/// The models a client trusts without being told otherwise.
///
/// A model file comes from the server and shapes how the query is processed on the client, so
/// one built to behave differently for certain words could reveal them through a crash or a
/// delay, without touching the encryption. Refusing every file not listed here closes that off.
pub const KNOWN_MODELS: [KnownModel; 2] = [
    KnownModel {
        key: "minilm",
        file: "all-MiniLM-L6-v2.usm",
        sha256: "b3fd40c159f2dca22fdeba20ed584e64ed980208be96486068f3789d8c32b51f",
    },
    KnownModel {
        key: "bge",
        file: "bge-small-en-v1.5.usm",
        sha256: "eefe4ec54fc8f2d16dff877213276a081f64de9b44c3931bf4b1139894d0395d",
    },
];

/// Whether a packed model with this hash is on the trusted list.
pub fn is_known_model(sha256: &str) -> bool {
    KNOWN_MODELS.iter().any(|model| model.sha256 == sha256)
}

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

    /// Number of values, or `None` if the shape overflows.
    fn count(&self) -> Option<usize> {
        self.shape
            .iter()
            .try_fold(1usize, |total, &side| total.checked_mul(side))
    }

    /// Bytes this tensor occupies in the file, or `None` if the shape makes no sense. A packed
    /// tensor needs a positive number of rows of positive length.
    fn bytes(&self) -> Option<usize> {
        let count = self.count()?;
        if self.format == WeightFormat::F32 {
            return count.checked_mul(4);
        }
        if self.rows() == 0 || count == 0 {
            return None;
        }
        let scales = self.rows().checked_mul(4)?;
        let values = match self.format {
            WeightFormat::Q16 => count.checked_mul(2)?,
            _ => count,
        };
        scales.checked_add(values)
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
            let bytes = info
                .bytes()
                .ok_or_else(|| err(format!("tensor {} has an impossible shape", info.name)))?;
            let end = offset
                .checked_add(bytes)
                .filter(|&end| end <= self.data.len())
                .ok_or_else(|| err("model file is truncated"))?;
            if info.name == name {
                return Ok((decode(info, &self.data[offset..end]), &info.shape));
            }
            offset = end;
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
        if outputs == 0 || inputs == 0 {
            return Err(err(format!("{prefix} has no width")));
        }
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

/// Expand a tensor to floats. `raw` must be exactly `info.bytes()` long, which also guarantees
/// that a packed tensor has a positive number of rows of positive length.
fn decode(info: &TensorInfo, raw: &[u8]) -> Vec<f32> {
    if info.format == WeightFormat::F32 {
        return floats(raw);
    }
    let rows = info.rows();
    let count = info.count().unwrap_or(0);
    let row_len = count / rows;
    let (scales, body) = raw.split_at(rows * 4);
    let scales = floats(scales);
    let mut out = Vec::with_capacity(count);
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
        if header.hidden == 0 || header.hidden > MAX_HIDDEN {
            return Err(err("the vector length is out of range"));
        }
        if header.heads == 0 || header.hidden % header.heads != 0 {
            return Err(err("hidden size is not divisible by the head count"));
        }
        // Every layer needs tensors in the file, so a missing one ends this loop early.
        let mut layers = Vec::new();
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
        let embed_norm = t.norm("embeddings.LayerNorm")?;
        // A hostile model file must fail here and not panic later while encoding. A panic that
        // happens only for some queries would tell the server something about them.
        let h = header.hidden;
        let norm = |n: &LayerNorm| n.gamma.len() == h && n.beta.len() == h;
        let shapes_ok = word_rows.checked_mul(h) == Some(word.len())
            && position_rows.checked_mul(h) == Some(position.len())
            && word_rows >= header.vocab.len()
            && position_rows >= header.max_tokens
            && header.max_tokens >= 2
            && token_types.len() >= h
            && norm(&embed_norm)
            && layers.iter().all(|l| {
                let square = |x: &Linear| x.inputs == h && x.outputs == h;
                let fits = |x: &Linear| {
                    x.inputs.checked_mul(x.outputs) == Some(x.weight.len())
                        && x.bias.len() == x.outputs
                };
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
            embed_norm,
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

    /// Embed a search query, keeping at most 128 tokens of it.
    pub fn embed_query(&self, text: &str) -> Vec<f32> {
        self.bert.encode(&self.query_tokens(text))
    }

    fn query_tokens(&self, text: &str) -> Vec<u32> {
        let limit = MAX_QUERY_TOKENS.min(self.max_tokens);
        self.tokenizer
            .encode(&format!("{}{text}", self.query_prefix), limit)
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

    const VOCAB: [&str; 9] = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "the", "secret", "word", "##s", "a",
    ];
    const H: usize = 4;
    const FFN: usize = 8;

    fn float_tensor(header: &mut Vec<TensorInfo>, data: &mut Vec<u8>, name: &str, shape: &[usize]) {
        let count: usize = shape.iter().product();
        for i in 0..count {
            data.extend((0.01 * (i % 7) as f32 + 0.02).to_le_bytes());
        }
        header.push(TensorInfo {
            name: name.to_owned(),
            shape: shape.to_vec(),
            format: WeightFormat::F32,
        });
    }

    fn q8_tensor(header: &mut Vec<TensorInfo>, data: &mut Vec<u8>, name: &str, shape: &[usize]) {
        for _ in 0..shape[0] {
            data.extend(0.01f32.to_le_bytes());
        }
        for i in 0..shape.iter().product::<usize>() {
            data.push((i % 11) as u8);
        }
        header.push(TensorInfo {
            name: name.to_owned(),
            shape: shape.to_vec(),
            format: WeightFormat::Q8,
        });
    }

    /// A complete one-layer model, small enough to mutate thousands of times.
    fn tiny_model() -> (Header, Vec<u8>) {
        let (mut infos, mut data) = (Vec::new(), Vec::new());
        let linear = |infos: &mut Vec<_>, data: &mut Vec<_>, name: &str, o: usize, i: usize| {
            q8_tensor(infos, data, &format!("{name}.weight"), &[o, i]);
            float_tensor(infos, data, &format!("{name}.bias"), &[o]);
        };
        let norm = |infos: &mut Vec<_>, data: &mut Vec<_>, name: &str| {
            float_tensor(infos, data, &format!("{name}.weight"), &[H]);
            float_tensor(infos, data, &format!("{name}.bias"), &[H]);
        };
        let p = "encoder.layer.0";
        for part in ["query", "key", "value"] {
            let name = format!("{p}.attention.self.{part}");
            linear(&mut infos, &mut data, &name, H, H);
        }
        let name = format!("{p}.attention.output.dense");
        linear(&mut infos, &mut data, &name, H, H);
        let name = format!("{p}.attention.output.LayerNorm");
        norm(&mut infos, &mut data, &name);
        let name = format!("{p}.intermediate.dense");
        linear(&mut infos, &mut data, &name, FFN, H);
        let name = format!("{p}.output.dense");
        linear(&mut infos, &mut data, &name, H, FFN);
        let name = format!("{p}.output.LayerNorm");
        norm(&mut infos, &mut data, &name);
        let name = "embeddings.token_type_embeddings.weight";
        float_tensor(&mut infos, &mut data, name, &[2, H]);
        let name = "embeddings.word_embeddings.weight";
        float_tensor(&mut infos, &mut data, name, &[VOCAB.len(), H]);
        let name = "embeddings.position_embeddings.weight";
        float_tensor(&mut infos, &mut data, name, &[16, H]);
        norm(&mut infos, &mut data, "embeddings.LayerNorm");
        let header = Header {
            name: "tiny".into(),
            hidden: H,
            layers: 1,
            heads: 2,
            pooling: Pooling::Mean,
            lowercase: true,
            query_prefix: String::new(),
            document_prefix: String::new(),
            max_tokens: 16,
            vocab: VOCAB.iter().map(|&w| w.to_owned()).collect(),
            tensors: infos,
        };
        (header, data)
    }

    fn file(header: &Header, data: &[u8]) -> Vec<u8> {
        let json = serde_json::to_vec(header).unwrap();
        let mut out = MAGIC.to_vec();
        out.extend((json.len() as u32).to_le_bytes());
        out.extend(json);
        out.extend_from_slice(data);
        out
    }

    fn with_shape(name: &str, shape: &[usize]) -> Vec<u8> {
        let (mut header, data) = tiny_model();
        let tensor = header.tensors.iter_mut().find(|t| t.name == name).unwrap();
        tensor.shape = shape.to_vec();
        file(&header, &data)
    }

    #[test]
    fn the_tiny_model_works() {
        let (header, data) = tiny_model();
        let model = Embedder::load(&file(&header, &data)).unwrap();
        assert_eq!(model.dim(), H);
        assert_eq!(model.embed_query("the secret words").len(), H);
    }

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

    /// A shape whose byte count wraps around to the real length would pass a size check, and
    /// then a word given an id past the end of the table would panic, but only for queries
    /// containing that word. That silence is visible to the server, so it must never load.
    #[test]
    fn a_shape_that_wraps_around_is_refused() {
        let word = "embeddings.word_embeddings.weight";
        let wrap = (1usize << 62) + VOCAB.len();
        assert!(Embedder::load(&with_shape(word, &[wrap, H])).is_err());
        let wrap = (1usize << 57) + VOCAB.len();
        assert!(Embedder::load(&with_shape(word, &[wrap, H])).is_err());
        let position = "embeddings.position_embeddings.weight";
        assert!(Embedder::load(&with_shape(position, &[(1usize << 62) + 16, H])).is_err());
    }

    #[test]
    fn a_vocabulary_larger_than_the_word_table_is_refused() {
        let (mut header, data) = tiny_model();
        header.vocab.push("trigger".into());
        assert!(Embedder::load(&file(&header, &data)).is_err());
    }

    #[test]
    fn impossible_shapes_are_refused_without_panicking() {
        let dense = "encoder.layer.0.attention.self.query.weight";
        for shape in [
            &[5usize, 0][..],
            &[0, 4],
            &[0],
            &[1 << 40, (1 << 23) - 2],
            &[1 << 32, 1 << 32],
            &[usize::MAX, usize::MAX],
        ] {
            assert!(
                Embedder::load(&with_shape(dense, shape)).is_err(),
                "{shape:?}"
            );
        }
    }

    #[test]
    fn impossible_architectures_are_refused_without_panicking() {
        type Change = fn(&mut Header);
        let cases: [(&str, Change); 6] = [
            ("no width", |h| h.hidden = 0),
            ("huge width", |h| h.hidden = 1 << 40),
            ("no heads", |h| h.heads = 0),
            ("heads that do not divide", |h| h.heads = 3),
            ("more layers than tensors", |h| h.layers = 1 << 60),
            ("no positions for the input", |h| h.max_tokens = 1 << 40),
        ];
        for (label, change) in cases {
            let (mut header, data) = tiny_model();
            change(&mut header);
            assert!(Embedder::load(&file(&header, &data)).is_err(), "{label}");
        }
    }

    #[test]
    fn an_empty_norm_is_refused_rather_than_failing_while_encoding() {
        let (mut header, mut data) = tiny_model();
        // Rebuild with the final norm empty, keeping the rest of the file consistent.
        for name in ["embeddings.LayerNorm.weight", "embeddings.LayerNorm.bias"] {
            let tensor = header.tensors.iter_mut().find(|t| t.name == name).unwrap();
            tensor.shape = vec![0];
        }
        data.truncate(data.len() - 2 * H * 4);
        assert!(Embedder::load(&file(&header, &data)).is_err());
    }

    /// Set these tensors to the given shapes, stored as floats, and rebuild the data to match.
    fn reshaped(changes: &[(&str, &[usize])]) -> Vec<u8> {
        let (mut header, _) = tiny_model();
        let mut data = Vec::new();
        let infos = std::mem::take(&mut header.tensors);
        for info in infos {
            let changed = changes.iter().find(|(name, _)| *name == info.name);
            let shape = changed
                .map_or(info.shape.as_slice(), |(_, shape)| shape)
                .to_vec();
            float_tensor(&mut header.tensors, &mut data, &info.name, &shape);
        }
        file(&header, &data)
    }

    /// Every width is consistent here, so only the check for a zero width can refuse this.
    #[test]
    fn a_zero_width_layer_is_refused_rather_than_failing_while_encoding() {
        let p = "encoder.layer.0";
        let consistent = reshaped(&[
            (&format!("{p}.intermediate.dense.weight"), &[0, H]),
            (&format!("{p}.intermediate.dense.bias"), &[0]),
            (&format!("{p}.output.dense.weight"), &[H, 0]),
        ]);
        assert!(Embedder::load(&consistent).is_err());
    }

    /// With no layers and tensors of no width, everything else about this model is consistent.
    #[test]
    fn a_model_of_no_width_is_refused_rather_than_failing_while_encoding() {
        let (mut header, _) = tiny_model();
        header.hidden = 0;
        header.layers = 0;
        header.tensors.retain(|t| t.name.starts_with("embeddings."));
        for tensor in &mut header.tensors {
            tensor.format = WeightFormat::F32;
            *tensor.shape.last_mut().unwrap() = 0;
        }
        assert!(Embedder::load(&file(&header, &[])).is_err());
    }

    #[test]
    fn queries_are_cut_to_a_fixed_length() {
        let (mut header, _) = tiny_model();
        header.max_tokens = 400;
        let mut data = Vec::new();
        let infos = std::mem::take(&mut header.tensors);
        for info in infos {
            let shape = if info.name == "embeddings.position_embeddings.weight" {
                vec![400, H]
            } else {
                info.shape.clone()
            };
            float_tensor(&mut header.tensors, &mut data, &info.name, &shape);
        }
        let model = Embedder::load(&file(&header, &data)).unwrap();
        let long = "secret ".repeat(1000);
        assert_eq!(model.query_tokens(&long).len(), MAX_QUERY_TOKENS);
        assert_eq!(model.embed_query(&long).len(), H);
    }

    /// Whatever a damaged file does, loading must return, and anything that loads must then
    /// encode every kind of input without panicking.
    #[test]
    fn random_damage_never_causes_a_panic() {
        let (header, data) = tiny_model();
        let original = file(&header, &data);
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let queries = ["", "the secret words", "secret ", &"a ".repeat(200)];
        let mut loaded = 0;
        for _ in 0..4000 {
            let mut bytes = original.clone();
            for _ in 0..1 + next() % 4 {
                let at = next() as usize % bytes.len();
                bytes[at] = match next() % 3 {
                    0 => (next() % 256) as u8,
                    1 => b'0' + (next() % 10) as u8,
                    _ => bytes[at] ^ (1 << (next() % 8)),
                };
            }
            if let Ok(model) = Embedder::load(&bytes) {
                loaded += 1;
                for query in queries {
                    assert_eq!(model.embed_query(query).len(), model.dim());
                }
            }
        }
        assert!(
            loaded > 0,
            "no damaged file loaded, so encoding was never tried"
        );
    }

    #[test]
    fn only_listed_models_are_known() {
        assert!(is_known_model(KNOWN_MODELS[0].sha256));
        assert!(!is_known_model("00"));
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
