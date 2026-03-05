//! Pure Opus encoder/decoder using `audiopus` (libopus binding).
//!
//! This module provides Opus encoding/decoding without depending on ffmpeg.
//! It uses the `audiopus` crate which wraps just the libopus C library —
//! a small, portable library that cross-compiles easily for iOS and Android,
//! unlike the full ffmpeg build.
//!
//! The encoded bitstream is fully compatible with the ffmpeg-based Opus
//! encoder/decoder since both use libopus under the hood.

mod decoder;
mod encoder;

pub use decoder::PureOpusDecoder;
pub use encoder::PureOpusEncoder;

/// Audio-only decoders bundle (no video).
#[derive(Debug, Clone, Copy)]
pub struct AudioOnlyDecoders;
