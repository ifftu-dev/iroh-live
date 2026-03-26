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

//! iOS-only video encoder/decoder/capture using Apple VideoToolbox + AVFoundation.
//!
//! This module is gated behind `feature = "video-ios"` and provides:
//!
//! - [`VtEncoder`]: H.264 encoder using `VTCompressionSession` (BGRA → H.264 length-prefixed NALUs)
//! - [`VtDecoder`]: H.264 decoder using `VTDecompressionSession` (H.264 → RGBA)
//! - [`IosCameraSource`]: Camera capture using `AVCaptureSession` (→ BGRA `VideoFrame`s)
//! - [`IosDecoders`]: Bundle struct implementing [`Decoders`] with `PureOpusDecoder` + `VtDecoder`

pub mod camera;
mod decoder;
mod encoder;

pub use camera::IosCameraSource;
pub use decoder::VtDecoder;
pub use encoder::VtEncoder;

use crate::av::{AudioDecoder, Decoders, VideoDecoder};
use crate::opus::PureOpusDecoder;

/// Decoder bundle for iOS: pure-Rust Opus audio + VideoToolbox H.264 video.
pub struct IosDecoders;

impl Decoders for IosDecoders {
    type Audio = PureOpusDecoder;
    type Video = VtDecoder;
}
