// Copyright 2025 N0, INC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


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
