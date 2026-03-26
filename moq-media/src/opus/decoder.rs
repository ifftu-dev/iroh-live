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

//! Pure Opus decoder using `audiopus` (libopus binding, no ffmpeg).

use anyhow::Result;
use audiopus::{Channels, SampleRate, coder::Decoder, packet::Packet};
use hang::catalog::AudioConfig;

use crate::av::{AudioDecoder, AudioFormat};

/// Maximum decoded frame size: 120ms at 48kHz stereo.
/// Opus can decode up to 120ms frames; 48000 * 0.120 * 2 = 11520 samples.
const MAX_FRAME_SAMPLES: usize = 48_000 * 120 / 1000 * 2;

pub struct PureOpusDecoder {
    decoder: Decoder,
    decode_buf: Vec<f32>,
    decoded_len: usize,
    target_channels: u32,
    source_channels: u32,
    target_sample_rate: u32,
    source_sample_rate: u32,
    converted_buf: Vec<f32>,
}

impl AudioDecoder for PureOpusDecoder {
    fn new(config: &AudioConfig, target_format: AudioFormat) -> Result<Self>
    where
        Self: Sized,
    {
        match config.codec {
            hang::catalog::AudioCodec::Opus => {}
            _ => anyhow::bail!(
                "Unsupported codec {} (only opus is supported)",
                config.codec
            ),
        }

        let sr = match config.sample_rate {
            8000 => SampleRate::Hz8000,
            12000 => SampleRate::Hz12000,
            16000 => SampleRate::Hz16000,
            24000 => SampleRate::Hz24000,
            48000 => SampleRate::Hz48000,
            other => anyhow::bail!("Unsupported sample rate {other} for Opus decoder"),
        };

        let channels = match config.channel_count {
            1 => Channels::Mono,
            2 => Channels::Stereo,
            other => anyhow::bail!("Unsupported channel count {other} for Opus decoder"),
        };

        let decoder = Decoder::new(sr, channels)
            .map_err(|e| anyhow::anyhow!("Failed to create Opus decoder: {e}"))?;

        tracing::info!(
            "Pure Opus decoder initialized: {}Hz {} ch -> {}Hz {} ch",
            config.sample_rate,
            config.channel_count,
            target_format.sample_rate,
            target_format.channel_count
        );

        Ok(Self {
            decoder,
            decode_buf: vec![0.0f32; MAX_FRAME_SAMPLES],
            decoded_len: 0,
            target_channels: target_format.channel_count,
            source_channels: config.channel_count,
            target_sample_rate: target_format.sample_rate,
            source_sample_rate: config.sample_rate,
            converted_buf: Vec::new(),
        })
    }

    fn push_packet(&mut self, packet: hang::Frame) -> Result<()> {
        let payload_bytes: Vec<u8> = {
            use bytes::Buf;
            let mut data = packet.payload;
            let mut bytes = vec![0u8; data.num_bytes()];
            data.copy_to_slice(&mut bytes);
            bytes
        };

        let opus_packet = Packet::try_from(&payload_bytes[..])
            .map_err(|e| anyhow::anyhow!("Invalid Opus packet: {e}"))?;

        let output = audiopus::MutSignals::try_from(&mut self.decode_buf[..])
            .map_err(|e| anyhow::anyhow!("Failed to create MutSignals: {e}"))?;

        let decoded_samples_per_channel = self
            .decoder
            .decode_float(Some(opus_packet), output, false)
            .map_err(|e| anyhow::anyhow!("Opus decode failed: {e}"))?;

        self.decoded_len = decoded_samples_per_channel * self.source_channels as usize;

        tracing::trace!(
            "push_packet: {} bytes -> {} samples ({} per ch)",
            payload_bytes.len(),
            self.decoded_len,
            decoded_samples_per_channel
        );

        Ok(())
    }

    fn pop_samples(&mut self) -> Result<Option<&[f32]>> {
        if self.decoded_len == 0 {
            return Ok(None);
        }

        let source_samples = &self.decode_buf[..self.decoded_len];

        if self.source_channels == self.target_channels
            && self.source_sample_rate == self.target_sample_rate
        {
            self.decoded_len = 0;
            return Ok(Some(source_samples));
        }

        match (self.source_channels, self.target_channels) {
            (2, 1) => {
                let mono_samples = source_samples.len() / 2;
                self.converted_buf.clear();
                self.converted_buf.reserve(mono_samples);
                for i in 0..mono_samples {
                    let left = source_samples[i * 2];
                    let right = source_samples[i * 2 + 1];
                    self.converted_buf.push((left + right) * 0.5);
                }
                self.decoded_len = 0;
                Ok(Some(&self.converted_buf))
            }
            (1, 2) => {
                self.converted_buf.clear();
                self.converted_buf.reserve(source_samples.len() * 2);
                for &sample in source_samples {
                    self.converted_buf.push(sample);
                    self.converted_buf.push(sample);
                }
                self.decoded_len = 0;
                Ok(Some(&self.converted_buf))
            }
            _ => {
                self.decoded_len = 0;
                Ok(Some(source_samples))
            }
        }
    }
}
