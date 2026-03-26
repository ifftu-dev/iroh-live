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

//! Pure Opus encoder using `audiopus` (libopus binding, no ffmpeg).

use anyhow::Result;
use audiopus::{Application, Channels, SampleRate, coder::Encoder};
use hang::{Timestamp, catalog::AudioConfig};
use tracing::trace;

use crate::av::{AudioEncoder, AudioEncoderInner, AudioFormat, AudioPreset};

const SAMPLE_RATE: u32 = 48_000;
const BITRATE: u64 = 128_000; // 128 kbps

/// Maximum Opus packet size in bytes (recommended by Opus docs).
const MAX_PACKET_SIZE: usize = 4000;

/// Opus frame size in samples per channel at 48kHz for 20ms frames.
const FRAME_SIZE: usize = 960;

pub struct PureOpusEncoder {
    encoder: Encoder,
    sample_count: u64,
    sample_rate: u32,
    bitrate: u64,
    channel_count: u32,
    /// Buffer used by the Opus encoder to write encoded output into.
    /// Always kept at `MAX_PACKET_SIZE` capacity; the actual encoded
    /// length from the last `push_samples` call is stored in `encoded_len`.
    encode_buf: Vec<u8>,
    /// Number of valid encoded bytes in `encode_buf` after the last
    /// `push_samples` call, or `None` if no data is ready to pop.
    encoded_len: Option<usize>,
}

impl PureOpusEncoder {
    pub fn stereo() -> Result<Self> {
        Self::new(SAMPLE_RATE, 2, BITRATE)
    }

    pub fn mono() -> Result<Self> {
        Self::new(SAMPLE_RATE, 1, BITRATE)
    }

    pub fn new(sample_rate: u32, channel_count: u32, bitrate: u64) -> Result<Self> {
        tracing::info!(
            "Initializing pure Opus encoder: {}Hz, {} channels, {} bps",
            sample_rate,
            channel_count,
            bitrate
        );

        let sr = match sample_rate {
            8000 => SampleRate::Hz8000,
            12000 => SampleRate::Hz12000,
            16000 => SampleRate::Hz16000,
            24000 => SampleRate::Hz24000,
            48000 => SampleRate::Hz48000,
            _ => anyhow::bail!("Unsupported sample rate {sample_rate} for Opus"),
        };

        let channels = match channel_count {
            1 => Channels::Mono,
            2 => Channels::Stereo,
            _ => anyhow::bail!("Unsupported channel count {channel_count} for Opus"),
        };

        let mut encoder = Encoder::new(sr, channels, Application::Voip)
            .map_err(|e| anyhow::anyhow!("Failed to create Opus encoder: {e}"))?;

        // Set bitrate
        let opus_bitrate = if bitrate <= 512_000 {
            audiopus::Bitrate::BitsPerSecond(bitrate as i32)
        } else {
            audiopus::Bitrate::Max
        };
        encoder
            .set_bitrate(opus_bitrate)
            .map_err(|e| anyhow::anyhow!("Failed to set bitrate: {e}"))?;

        // Enable VBR for better quality
        encoder
            .set_vbr(true)
            .map_err(|e| anyhow::anyhow!("Failed to enable VBR: {e}"))?;

        // Set complexity to 10 for best quality
        encoder
            .set_complexity(10)
            .map_err(|e| anyhow::anyhow!("Failed to set complexity: {e}"))?;

        // Enable inband FEC for network resilience
        encoder
            .set_inband_fec(true)
            .map_err(|e| anyhow::anyhow!("Failed to enable inband FEC: {e}"))?;

        tracing::info!("Pure Opus encoder initialized successfully");

        Ok(Self {
            encoder,
            sample_count: 0,
            sample_rate,
            channel_count,
            bitrate,
            encode_buf: vec![0u8; MAX_PACKET_SIZE],
            encoded_len: None,
        })
    }
}

impl AudioEncoder for PureOpusEncoder {
    fn with_preset(format: AudioFormat, preset: AudioPreset) -> Result<Self>
    where
        Self: Sized,
    {
        let channel_count = format.channel_count;
        let bitrate = match preset {
            AudioPreset::Hq => BITRATE,
            AudioPreset::Lq => 32_000,
        };
        Self::new(SAMPLE_RATE, channel_count, bitrate)
    }
}

impl AudioEncoderInner for PureOpusEncoder {
    fn name(&self) -> &str {
        "opus"
    }

    fn config(&self) -> AudioConfig {
        hang::catalog::AudioConfig {
            codec: hang::catalog::AudioCodec::Opus,
            sample_rate: self.sample_rate,
            channel_count: self.channel_count,
            bitrate: Some(self.bitrate),
            // The pure opus encoder doesn't need extradata — the decoder
            // can initialize from the config fields alone. This is compatible
            // with the ffmpeg decoder which treats extradata as optional.
            description: None,
        }
    }

    fn push_samples(&mut self, samples: &[f32]) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }

        let samples_per_channel = samples.len() / self.channel_count as usize;
        debug_assert_eq!(
            samples_per_channel, FRAME_SIZE,
            "Expected {} samples per channel (20ms at 48kHz), got {}",
            FRAME_SIZE, samples_per_channel
        );

        self.encode_buf.resize(MAX_PACKET_SIZE, 0);
        let encoded_len = self
            .encoder
            .encode_float(samples, &mut self.encode_buf)
            .map_err(|e| anyhow::anyhow!("Opus encode failed: {e}"))?;

        trace!(
            "push_samples: {} samples -> {} bytes",
            samples.len(),
            encoded_len
        );
        self.sample_count += samples_per_channel as u64;
        self.encoded_len = Some(encoded_len);

        Ok(())
    }

    fn pop_packet(&mut self) -> Result<Option<hang::Frame>> {
        let len = match self.encoded_len.take() {
            Some(l) => l,
            None => return Ok(None),
        };

        let payload = self.encode_buf[..len].to_vec();

        let hang_frame = hang::Frame {
            payload: payload.into(),
            timestamp: Timestamp::from_micros(
                (self.sample_count * 1_000_000) / self.sample_rate as u64,
            )?,
            keyframe: true,
        };
        trace!("pop_packet: {} bytes", hang_frame.payload.num_bytes());
        Ok(Some(hang_frame))
    }
}
