//! Pictures in a state.
//!
//! A decision model that can see is shown a picture as one item of a state that is an array,
//! in the form OpenRouter's decisions take one:
//!
//! ```text
//! "state": [{"instructions": ..., "inputs": ...}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]
//! ```
//!
//! A text item of that array is a plain string; a picture inside an object is not read as a
//! picture. A picture here is always its bytes, in a data URL: never an address to fetch,
//! whose contents could change under a key ([`crate::key`] covers the bytes).

use serde_json::json;
use sha2::{Digest, Sha256};

use crate::wire::Json;

/// The kinds of picture a decision model is shown: PNG, JPEG and WebP. (Jev reads the first two.)
pub const CONTENT_TYPES: [&str; 3] = ["image/png", "image/jpeg", "image/webp"];

/// The `type` of a picture, as an item of a state.
pub const IMAGE_URL: &str = "image_url";

/// The `type` of what stands for a picture where its bytes were not kept: [`Image::reference`].
pub const IMAGE_REF: &str = "image_ref";

/// A picture as an item of a state: its bytes, as a data URL in OpenRouter's form. No file is
/// read and no clock: a module with nothing under it builds one the same.
///
/// ```
/// let part = systemone::image("image/png", b"\x89PNG");
/// assert_eq!(part, serde_json::json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw=="}}));
/// let picture = systemone::Image::of(&part).unwrap();
/// assert_eq!((picture.content_type, picture.bytes().unwrap()), ("image/png", b"\x89PNG".to_vec()));
/// ```
pub fn image(content_type: &str, bytes: &[u8]) -> Json {
    json!({"type": IMAGE_URL, "image_url": {"url": format!("data:{content_type};base64,{}", encode(bytes))}})
}

/// How many pictures a state shows: the items of an array state whose `type` is `image_url`,
/// as a decision model reads them. A picture inside an object is not one, and neither is any
/// state that is not an array.
///
/// ```
/// let state = serde_json::json!([{"inputs": {"window": "<image: window>"}}, systemone::image("image/png", b"png")]);
/// assert_eq!(systemone::images(&state), 1);
/// assert_eq!(systemone::images(&serde_json::json!({"inputs": {"window": systemone::image("image/png", b"png")}})), 0);
/// ```
pub fn images(state: &Json) -> usize {
    match state {
        Json::Array(items) => items.iter().filter(|item| is_image(item)).count(),
        _ => 0,
    }
}

/// Whether an item of an array state is a picture, as a decision model reads one: an object
/// whose `type` is `image_url`.
pub fn is_image(item: &Json) -> bool {
    item.get("type").and_then(Json::as_str) == Some(IMAGE_URL)
}

/// A picture as [`image`] makes one, read: what kind it is, and its bytes in base64.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Image<'a> {
    /// `image/png`, `image/jpeg` or `image/webp`.
    pub content_type: &'a str,
    /// The bytes, in base64.
    pub data: &'a str,
}

impl<'a> Image<'a> {
    /// The picture a value is, if it is one in the form [`image`] makes: `{"type": "image_url",
    /// "image_url": {"url": "data:<kind>;base64,<bytes>"}}` and nothing more, of one of the
    /// [`CONTENT_TYPES`], in base64 that reads. Anything else is not one.
    pub fn of(value: &'a Json) -> Option<Self> {
        let part = value.as_object().filter(|part| part.len() == 2)?;
        if part.get("type")?.as_str()? != IMAGE_URL {
            return None;
        }
        let address = part.get(IMAGE_URL)?.as_object().filter(|address| address.len() == 1)?;
        let url = address.get("url")?.as_str()?;
        let (content_type, data) = url.strip_prefix("data:")?.split_once(";base64,")?;
        let known = CONTENT_TYPES.contains(&content_type);
        (known && valid(data)).then_some(Self { content_type, data })
    }

    /// The picture's bytes. None only for base64 that does not read, which [`Image::of`] never gives.
    pub fn bytes(&self) -> Option<Vec<u8>> {
        decode(self.data)
    }

    /// The SHA-256 of the picture's bytes, in hex.
    pub fn sha256(&self) -> String {
        let bytes = self.bytes().unwrap_or_default();
        Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// What stands for the picture where its bytes are not kept: `{"type": "image_ref",
    /// "sha256": ..., "content_type": ...}`, the hash of its bytes.
    ///
    /// ```
    /// let part = systemone::image("image/png", b"");
    /// let kept = systemone::Image::of(&part).unwrap().reference();
    /// assert_eq!(kept["sha256"], "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    /// assert_eq!((kept["type"].as_str(), kept["content_type"].as_str()), (Some("image_ref"), Some("image/png")));
    /// ```
    pub fn reference(&self) -> Json {
        json!({"type": IMAGE_REF, "sha256": self.sha256(), "content_type": self.content_type})
    }
}

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Bytes in standard base64, padded.
pub(crate) fn encode(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let group = chunk
            .iter()
            .enumerate()
            .fold(0u32, |group, (at, byte)| group | (u32::from(*byte) << (16 - 8 * at)));
        for position in 0..4 {
            if position <= chunk.len() {
                text.push(ALPHABET[((group >> (18 - 6 * position)) & 63) as usize] as char);
            } else {
                text.push('=');
            }
        }
    }
    text
}

/// The value of each byte as a digit of base64, by the byte: [`NOT_A_DIGIT`] for any that is not one.
const VALUES: [u8; 256] = {
    let mut values = [NOT_A_DIGIT; 256];
    let mut digit = 0;
    while digit < 64 {
        values[ALPHABET[digit] as usize] = digit as u8;
        digit += 1;
    }
    values
};

/// What [`VALUES`] holds for a byte that is no digit of base64.
const NOT_A_DIGIT: u8 = 0xff;

/// Walk standard base64, padded, a group of four digits at a time: each group's bytes and how
/// many of them there are go to `each`. False, and the walk stopped, for anything else.
fn walk(text: &str, mut each: impl FnMut([u8; 3], usize)) -> bool {
    let text = text.as_bytes();
    if !text.len().is_multiple_of(4) {
        return false;
    }
    let groups = text.len() / 4;
    for (index, group) in text.as_chunks::<4>().0.iter().enumerate() {
        let padding = match (group[2], group[3]) {
            (b'=', b'=') => 2,
            (_, b'=') => 1,
            _ => 0,
        };
        if padding > 0 && index + 1 != groups {
            return false;
        }
        let mut bits = 0u32;
        for character in &group[..4 - padding] {
            let value = VALUES[*character as usize];
            if value == NOT_A_DIGIT {
                return false;
            }
            bits = (bits << 6) | u32::from(value);
        }
        bits <<= 6 * padding as u32;
        each([(bits >> 16) as u8, (bits >> 8) as u8, bits as u8], 3 - padding);
    }
    true
}

/// Whether a text is standard base64, padded. Nothing is decoded into memory.
pub(crate) fn valid(text: &str) -> bool {
    walk(text, |_, _| {})
}

/// The bytes of standard base64, padded: None for anything else.
pub(crate) fn decode(text: &str) -> Option<Vec<u8>> {
    let mut bytes = Vec::with_capacity(text.len() / 4 * 3);
    walk(text, |group, count| bytes.extend_from_slice(&group[..count])).then_some(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_is_written_and_read_as_the_standard_writes_it() {
        let cases: [(&[u8], &str); 7] = [
            (b"", ""),
            (b"f", "Zg=="),
            (b"fo", "Zm8="),
            (b"foo", "Zm9v"),
            (b"foob", "Zm9vYg=="),
            (b"fooba", "Zm9vYmE="),
            (b"\xff\xfe\x00\x10", "//4AEA=="),
        ];
        for (bytes, text) in cases {
            assert_eq!(encode(bytes), text);
            assert_eq!(decode(text).as_deref(), Some(bytes));
        }
        for wrong in ["Zg=", "Z===", "Z=g=", "Zg==Zg==", "Zm9v!A==", "Zm9", "Zm9v\u{e9}A=="] {
            assert!(!valid(wrong), "{wrong}");
            assert_eq!(decode(wrong), None, "{wrong}");
        }
    }

    #[test]
    fn only_a_picture_in_the_form_made_is_one() {
        let part = image("image/jpeg", b"jpeg");
        assert!(Image::of(&part).is_some());
        let not = [
            image("image/gif", b"gif"),
            json!({"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}),
            json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,***"}}),
            json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,Zg==", "detail": "low"}}),
            json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,Zg=="}, "note": 1}),
            json!("data:image/png;base64,Zg=="),
        ];
        for value in &not {
            assert!(Image::of(value).is_none(), "{value}");
        }
    }
}
