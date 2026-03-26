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

use anyhow::{Context, Result};
use ffmpeg_next::{
    self as ffmpeg, codec, codec::Id as CodecId, util::frame::video::Video as FfmpegFrame,
};

use crate::{
    av::{self, DecodeConfig, DecodedFrame, VideoDecoder},
    ffmpeg::{
        ext::{CodecContextExt, PacketExt},
        video::util::{Rescaler, StreamClock},
    },
};

pub struct FfmpegVideoDecoder {
    codec: ffmpeg::decoder::Video,
    rescaler: Rescaler,
    clock: StreamClock,
    decoded: FfmpegFrame,
    viewport_changed: Option<(u32, u32)>,
    last_timestamp: Option<hang::Timestamp>,
    /// True when no avcC extradata was set — packets need AVCC-to-Annex B conversion
    needs_avcc_to_annexb: bool,
    /// When needs_avcc_to_annexb is true, skip packets until we see the first keyframe
    /// with SPS/PPS. Pre-keyframe non-IDR packets produce garbage without reference frames.
    got_first_keyframe: bool,
}

impl VideoDecoder for FfmpegVideoDecoder {
    fn name(&self) -> &str {
        self.codec.id().name()
    }

    fn new(config: &hang::catalog::VideoConfig, playback_config: &DecodeConfig) -> Result<Self>
    where
        Self: Sized,
    {
        ffmpeg::init()?;

        // Build a decoder context for H.264 and attach extradata (e.g., avcC)
        let has_extradata = config.description.is_some();
        let codec = match &config.codec {
            hang::catalog::VideoCodec::H264(_meta) => {
                let codec =
                    codec::decoder::find(CodecId::H264).context("H.264 decoder not found")?;
                let mut ctx = codec::context::Context::new_with_codec(codec);
                if let Some(description) = &config.description {
                    ctx.set_extradata(&description)?;
                }
                ctx.decoder().video().unwrap()
            }
            hang::catalog::VideoCodec::AV1(_meta) => {
                let codec = codec::decoder::find(CodecId::AV1).context("AV1 decoder not found")?;
                let mut ctx = codec::context::Context::new_with_codec(codec);
                if let Some(description) = &config.description {
                    ctx.set_extradata(&description)?;
                }
                ctx.decoder().video().unwrap()
            }
            _ => anyhow::bail!(
                "Unsupported codec {} (only h264 and av1 are supported)",
                config.codec
            ),
        };
        let needs_avcc_to_annexb = !has_extradata;
        if needs_avcc_to_annexb {
            tracing::warn!(
                "ffmpeg decoder: no extradata/avcC — will convert AVCC packets to Annex B"
            );
        }
        let rescaler = Rescaler::new(playback_config.pixel_format.to_ffmpeg(), None)?;
        let clock = StreamClock::default();
        let decoded = FfmpegFrame::empty();
        Ok(Self {
            codec,
            rescaler,
            clock,
            decoded,
            viewport_changed: None,
            last_timestamp: None,
            needs_avcc_to_annexb,
            got_first_keyframe: false,
        })
    }

    fn set_viewport(&mut self, w: u32, h: u32) {
        self.viewport_changed = Some((w, h));
    }

    fn push_packet(&mut self, packet: hang::Frame) -> Result<()> {
        if self.needs_avcc_to_annexb {
            // First, flatten payload into an ffmpeg packet (contiguous bytes)
            let raw_packet = packet.payload.to_ffmpeg_packet();
            let avcc_data = raw_packet.data().unwrap_or(&[]);

            // Log NAL unit types in the AVCC data for debugging
            let nal_types = parse_avcc_nal_types(avcc_data);

            // Skip non-keyframe packets before we get SPS/PPS from the first keyframe.
            // Without parameter sets, the decoder produces garbage reference frames that
            // corrupt all subsequent decoding.
            if !self.got_first_keyframe {
                let has_sps = nal_types.iter().any(|t| *t == "SPS");
                if !has_sps {
                    tracing::debug!(
                        "ffmpeg decoder: skipping pre-keyframe packet ({} bytes, NAL types: {:?})",
                        avcc_data.len(),
                        nal_types,
                    );
                    return Ok(());
                }
                tracing::info!(
                    "ffmpeg decoder: first keyframe received ({} bytes, NAL types: {:?})",
                    avcc_data.len(),
                    nal_types,
                );
                self.got_first_keyframe = true;
            }

            if packet.keyframe {
                tracing::info!(
                    "ffmpeg decoder: keyframe {} bytes, NAL types: {:?}",
                    avcc_data.len(),
                    nal_types,
                );
            }

            // Convert AVCC (4-byte length-prefixed NALs) to Annex B (start-code NALs)
            let annexb = avcc_to_annexb(avcc_data);

            tracing::debug!("ffmpeg decoder: Annex B output {} bytes", annexb.len(),);

            let mut ffmpeg_packet = ffmpeg::Packet::new(annexb.len());
            ffmpeg_packet.data_mut().unwrap().copy_from_slice(&annexb);
            self.codec
                .send_packet(&ffmpeg_packet)
                .context("ffmpeg decoder: failed to decode Annex B converted packet")?;
        } else {
            let ffmpeg_packet = packet.payload.to_ffmpeg_packet();
            self.codec.send_packet(&ffmpeg_packet)?;
        }
        self.last_timestamp = Some(packet.timestamp);
        Ok(())
    }

    fn pop_frame(&mut self) -> Result<Option<av::DecodedFrame>> {
        match self.codec.receive_frame(&mut self.decoded) {
            Ok(()) => {
                if self.viewport_changed.is_some() || self.rescaler.ctx.is_none() {
                    tracing::info!(
                        "ffmpeg decoder: raw={}x{} fmt={:?} stride={}, target={}x{} fmt={:?}",
                        self.decoded.width(),
                        self.decoded.height(),
                        self.decoded.format(),
                        self.decoded.stride(0),
                        self.rescaler
                            .target_width_height
                            .map(|(w, _)| w)
                            .unwrap_or(0),
                        self.rescaler
                            .target_width_height
                            .map(|(_, h)| h)
                            .unwrap_or(0),
                        self.rescaler.target_format,
                    );
                }

                if let Some((max_width, max_height)) = self.viewport_changed.take() {
                    let (width, height) =
                        calculate_resized_size(&self.decoded, max_width, max_height);
                    self.rescaler.set_target_dimensions(width, height);
                }

                let frame = self.rescaler.process(&mut self.decoded)?;
                let last_timestamp = self
                    .last_timestamp
                    .as_ref()
                    .context("missing last packet")?;
                let frame = DecodedFrame::from_ffmpeg(
                    frame,
                    self.clock.frame_delay(&last_timestamp),
                    std::time::Duration::from(*last_timestamp),
                );
                Ok(Some(frame))
            }
            Err(ffmpeg::util::error::Error::BufferTooSmall) => Ok(None),
            Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::util::error::EAGAIN => Ok(None),
            Err(err) => {
                // tracing::warn!("decoder error: {err} {err:?} {err:#?}");
                // Ok(None)
                Err(err.into())
            }
        }
    }
}

/// Parse AVCC-format data and return the NAL unit types present.
/// Used for diagnostic logging to verify SPS/PPS presence.
fn parse_avcc_nal_types(data: &[u8]) -> Vec<&'static str> {
    let mut types = Vec::new();
    let mut pos = 0;
    while pos + 4 <= data.len() {
        let nal_len =
            u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if nal_len == 0 || pos + nal_len > data.len() {
            break;
        }
        if nal_len > 0 {
            let nal_type = data[pos] & 0x1F;
            types.push(match nal_type {
                1 => "non-IDR",
                5 => "IDR",
                6 => "SEI",
                7 => "SPS",
                8 => "PPS",
                9 => "AUD",
                _ => "other",
            });
        }
        pos += nal_len;
    }
    types
}

/// Convert AVCC-format H.264 data (4-byte length-prefixed NAL units)
/// to Annex B format (start-code-prefixed NAL units).
///
/// This is needed when the decoder has no avcC extradata (out-of-band SPS/PPS),
/// which is the case when VideoToolbox encoder hasn't yet provided its
/// codec description through the catalog.
fn avcc_to_annexb(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 64);
    let mut pos = 0;
    while pos + 4 <= data.len() {
        let nal_len =
            u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if nal_len == 0 || pos + nal_len > data.len() {
            break;
        }
        // Annex B start code
        out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        out.extend_from_slice(&data[pos..pos + nal_len]);
        pos += nal_len;
    }
    out
}

/// Calculates the target frame size to fit into the requested bounds while preserving aspect ratio.
fn calculate_resized_size(decoded: &FfmpegFrame, max_width: u32, max_height: u32) -> (u32, u32) {
    let src_w = decoded.width().max(1);
    let src_h = decoded.height().max(1);
    let max_w = max_width.max(1);
    let max_h = max_height.max(1);

    // Fit within requested bounds, preserve aspect ratio, never upscale
    let scale_w = (max_w as f32) / (src_w as f32);
    let scale_h = (max_h as f32) / (src_h as f32);
    let scale = scale_w.min(scale_h).min(1.0).max(0.0);
    let target_width = ((src_w as f32) * scale).floor().max(1.0) as u32;
    let target_height = ((src_h as f32) * scale).floor().max(1.0) as u32;
    tracing::debug!(
        src_w,
        src_h,
        max_w,
        max_h,
        target_width,
        target_height,
        "scale"
    );
    (target_width, target_height)
}
