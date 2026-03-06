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
