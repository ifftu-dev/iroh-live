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

//! VideoToolbox H.264 decoder for iOS.
//!
//! Uses `VTDecompressionSession` to decode H.264 length-prefixed NAL units
//! back into RGBA frames (as `DecodedFrame` with `image::Frame`).

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use bytes::Bytes;

use crate::av::{DecodeConfig, DecodedFrame, PixelFormat, VideoDecoder};

// ── C types (same as encoder + VTDecompressionSession) ──

type CFAllocatorRef = *const c_void;
type CFDictionaryRef = *const c_void;
type CFMutableDictionaryRef = *mut c_void;
type CFStringRef = *const c_void;
type CFNumberRef = *const c_void;
type CFTypeRef = *const c_void;
type CMSampleBufferRef = *const c_void;
type CMFormatDescriptionRef = *const c_void;
type CMBlockBufferRef = *const c_void;
type CVPixelBufferRef = *const c_void;
type CVImageBufferRef = *const c_void;
type VTDecompressionSessionRef = *mut c_void;
type OSStatus = i32;

#[repr(C)]
#[derive(Copy, Clone)]
struct CMTimeRepr {
    value: i64,
    timescale: i32,
    flags: u32,
    epoch: i64,
}

const K_CM_TIME_FLAGS_VALID: u32 = 1;
const K_CV_PIXEL_FORMAT_TYPE_32_BGRA: u32 = 0x42475241;
const K_CV_PIXEL_FORMAT_TYPE_32_RGBA: u32 = 0x52474241; // 'RGBA' — not universally supported
const K_CF_NUMBER_SINT32_TYPE: i64 = 3;

#[repr(C)]
#[derive(Copy, Clone)]
struct CMSampleTimingInfo {
    duration: CMTimeRepr,
    presentation_time_stamp: CMTimeRepr,
    decode_time_stamp: CMTimeRepr,
}

fn cm_time(value: i64, timescale: i32) -> CMTimeRepr {
    CMTimeRepr {
        value,
        timescale,
        flags: K_CM_TIME_FLAGS_VALID,
        epoch: 0,
    }
}

const K_CM_TIME_INVALID: CMTimeRepr = CMTimeRepr {
    value: 0,
    timescale: 0,
    flags: 0,
    epoch: 0,
};

unsafe extern "C" {
    // CoreFoundation
    fn CFDictionaryCreateMutable(
        allocator: CFAllocatorRef,
        capacity: isize,
        key_callbacks: *const c_void,
        value_callbacks: *const c_void,
    ) -> CFMutableDictionaryRef;
    fn CFDictionarySetValue(dict: CFMutableDictionaryRef, key: *const c_void, value: *const c_void);
    fn CFRelease(cf: CFTypeRef);
    fn CFNumberCreate(
        allocator: CFAllocatorRef,
        the_type: i64,
        value_ptr: *const c_void,
    ) -> CFNumberRef;

    static kCFAllocatorDefault: CFAllocatorRef;
    static kCFAllocatorNull: CFAllocatorRef;
    static kCFTypeDictionaryKeyCallBacks: c_void;
    static kCFTypeDictionaryValueCallBacks: c_void;

    // CoreVideo
    static kCVPixelBufferPixelFormatTypeKey: CFStringRef;
    fn CVPixelBufferLockBaseAddress(pb: CVPixelBufferRef, flags: u64) -> OSStatus;
    fn CVPixelBufferUnlockBaseAddress(pb: CVPixelBufferRef, flags: u64) -> OSStatus;
    fn CVPixelBufferGetBaseAddress(pb: CVPixelBufferRef) -> *const c_void;
    fn CVPixelBufferGetBytesPerRow(pb: CVPixelBufferRef) -> usize;
    fn CVPixelBufferGetWidth(pb: CVPixelBufferRef) -> usize;
    fn CVPixelBufferGetHeight(pb: CVPixelBufferRef) -> usize;

    // CoreMedia
    fn CMVideoFormatDescriptionCreateFromH264ParameterSets(
        allocator: CFAllocatorRef,
        parameter_set_count: usize,
        parameter_set_pointers: *const *const u8,
        parameter_set_sizes: *const usize,
        nal_unit_header_length: i32,
        format_description_out: *mut CMFormatDescriptionRef,
    ) -> OSStatus;
    fn CMSampleBufferCreateReady(
        allocator: CFAllocatorRef,
        data_buffer: CMBlockBufferRef,
        format_description: CMFormatDescriptionRef,
        num_samples: i32,
        num_sample_timing_entries: i32,
        sample_timing_array: *const CMSampleTimingInfo,
        num_sample_size_entries: i32,
        sample_size_array: *const usize,
        sample_buffer_out: *mut CMSampleBufferRef,
    ) -> OSStatus;
    fn CMBlockBufferCreateWithMemoryBlock(
        allocator: CFAllocatorRef,
        memory_block: *const c_void,
        block_length: usize,
        block_allocator: CFAllocatorRef,
        custom_block_source: *const c_void,
        offset_to_data: usize,
        data_length: usize,
        flags: u32,
        block_buffer_out: *mut CMBlockBufferRef,
    ) -> OSStatus;

    // VideoToolbox
    fn VTDecompressionSessionCreate(
        allocator: CFAllocatorRef,
        video_format_description: CMFormatDescriptionRef,
        video_decoder_specification: CFDictionaryRef,
        destination_image_buffer_attributes: CFDictionaryRef,
        output_callback: *const VTDecompressionOutputCallbackRecord,
        decompression_session_out: *mut VTDecompressionSessionRef,
    ) -> OSStatus;
    fn VTDecompressionSessionDecodeFrame(
        session: VTDecompressionSessionRef,
        sample_buffer: CMSampleBufferRef,
        decode_flags: u32,
        source_frame_ref_con: *mut c_void,
        info_flags_out: *mut u32,
    ) -> OSStatus;
    fn VTDecompressionSessionWaitForAsynchronousFrames(
        session: VTDecompressionSessionRef,
    ) -> OSStatus;
    fn VTDecompressionSessionInvalidate(session: VTDecompressionSessionRef);
}

#[repr(C)]
struct VTDecompressionOutputCallbackRecord {
    decompress_output_callback: unsafe extern "C" fn(
        *mut c_void,      // decompressionOutputRefCon
        *mut c_void,      // sourceFrameRefCon
        OSStatus,         // status
        u32,              // infoFlags
        CVImageBufferRef, // imageBuffer
        CMTimeRepr,       // presentationTimeStamp
        CMTimeRepr,       // presentationDuration
    ),
    decompress_output_ref_con: *mut c_void,
}

/// Decoded frame waiting to be consumed.
struct RawDecodedFrame {
    rgba_data: Vec<u8>,
    width: u32,
    height: u32,
    timestamp_us: u64,
}

/// VideoToolbox H.264 decoder for iOS.
///
/// Supports two initialization modes:
/// - **Eager**: description (avcC) is available in the catalog → session created immediately.
/// - **Deferred**: description is missing → session created on first keyframe using inline SPS/PPS.
pub struct VtDecoder {
    session: VTDecompressionSessionRef,
    format_desc: CMFormatDescriptionRef,
    output: Arc<Mutex<VecDeque<RawDecodedFrame>>>,
    viewport_w: u32,
    viewport_h: u32,
    frame_count: u64,
    initialized: bool,
}

unsafe impl Send for VtDecoder {}

impl Drop for VtDecoder {
    fn drop(&mut self) {
        if !self.session.is_null() {
            unsafe {
                VTDecompressionSessionInvalidate(self.session);
                CFRelease(self.session as CFTypeRef);
            }
            self.session = ptr::null_mut();
        }
        if !self.format_desc.is_null() {
            unsafe { CFRelease(self.format_desc as CFTypeRef) };
            self.format_desc = ptr::null();
        }
    }
}

impl VideoDecoder for VtDecoder {
    fn new(config: &hang::catalog::VideoConfig, _playback_config: &DecodeConfig) -> Result<Self> {
        let viewport_w = config.coded_width.unwrap_or(640);
        let viewport_h = config.coded_height.unwrap_or(480);

        match config.description.as_ref() {
            Some(desc_bytes) => {
                tracing::info!(
                    "VtDecoder: got description ({} bytes) from catalog, attempting eager init",
                    desc_bytes.len()
                );
                match Self::create_session(desc_bytes) {
                    Ok((session, format_desc, output)) => {
                        tracing::info!(
                            "VtDecoder created (eager): session={session:?}, {viewport_w}x{viewport_h}"
                        );
                        Ok(Self {
                            session,
                            format_desc,
                            output,
                            viewport_w,
                            viewport_h,
                            frame_count: 0,
                            initialized: true,
                        })
                    }
                    Err(e) => {
                        tracing::warn!(
                            "VtDecoder: eager init failed: {e:#}, deferring to first keyframe"
                        );
                        Ok(Self::uninit(viewport_w, viewport_h))
                    }
                }
            }
            None => {
                tracing::warn!("VtDecoder: no description in catalog - will need keyframe to init");
                Ok(Self::uninit(viewport_w, viewport_h))
            }
        }
    }

    fn name(&self) -> &str {
        "vt-h264-dec"
    }

    fn push_packet(&mut self, packet: hang::Frame) -> Result<()> {
        if !self.initialized {
            if !packet.keyframe {
                return Ok(());
            }
            let raw_data: Vec<u8> = packet
                .payload
                .iter()
                .flat_map(|b| b.iter().copied())
                .collect();
            if raw_data.is_empty() {
                return Ok(());
            }
            tracing::info!(
                "VtDecoder: deferred init got first keyframe ({} bytes)",
                raw_data.len(),
            );
            let parse_result = if is_annex_b(&raw_data) {
                tracing::info!("VtDecoder: deferred init keyframe appears to be Annex B");
                parse_annex_b(&raw_data)
            } else {
                tracing::info!("VtDecoder: deferred init keyframe appears to be AVCC");
                self.try_extract_sps_pps_from_avcc(&raw_data)
            };
            let (sps, pps) = parse_result
                .map(|(sps, pps)| {
                    tracing::info!(
                        "VtDecoder: deferred init parsed SPS/PPS successfully (sps={}B, pps={}B)",
                        sps.len(),
                        pps.len()
                    );
                    (sps, pps)
                })
                .inspect_err(|e| {
                    tracing::warn!("VtDecoder: deferred init SPS/PPS parse failed: {e:#}");
                })?;
            let mut avcc = Vec::with_capacity(11 + sps.len() + pps.len());
            avcc.push(1); // version
            avcc.push(sps.get(1).copied().unwrap_or(0x42)); // profile
            avcc.push(sps.get(2).copied().unwrap_or(0xE0)); // compat
            avcc.push(sps.get(3).copied().unwrap_or(0x1F)); // level
            avcc.push(0xFF); // 4-byte NALU length
            avcc.push(0xE1); // 1 SPS
            avcc.extend_from_slice(&(sps.len() as u16).to_be_bytes());
            avcc.extend_from_slice(&sps);
            avcc.push(1); // 1 PPS
            avcc.extend_from_slice(&(pps.len() as u16).to_be_bytes());
            avcc.extend_from_slice(&pps);

            match Self::create_session(&avcc) {
                Ok((session, format_desc, output)) => {
                    tracing::info!("VtDecoder: deferred init succeeded, session={session:?}");
                    self.session = session;
                    self.format_desc = format_desc;
                    self.output = output;
                    self.initialized = true;
                }
                Err(e) => {
                    tracing::warn!("VtDecoder: deferred init failed: {e:#}");
                    return Ok(());
                }
            }
        }

        let raw_data: Vec<u8> = packet
            .payload
            .iter()
            .flat_map(|b| b.iter().copied())
            .collect();
        if raw_data.is_empty() {
            return Ok(());
        }

        self.frame_count += 1;
        if self.frame_count <= 10 || self.frame_count % 100 == 0 {
            let preview: String = raw_data
                .iter()
                .take(16)
                .map(|b| format!("{b:02x}"))
                .collect::<Vec<_>>()
                .join(" ");
            tracing::info!(
                "VtDecoder::push_packet #{}: {} bytes, keyframe={}, first 16: {preview}",
                self.frame_count,
                raw_data.len(),
                packet.keyframe
            );
        }

        // Detect whether the data uses Annex B start codes (00 00 00 01 or 00 00 01)
        // or is already length-prefixed (4-byte big-endian length + NAL data).
        // VTDecompressionSession requires length-prefixed NALUs.
        let data = if is_annex_b(&raw_data) {
            let converted = annex_b_to_length_prefixed(&raw_data);
            if self.frame_count <= 5 {
                tracing::info!(
                    "VtDecoder: converted Annex B ({} bytes) -> length-prefixed ({} bytes)",
                    raw_data.len(),
                    converted.len()
                );
            }
            converted
        } else {
            raw_data
        };
        if data.is_empty() {
            tracing::debug!(
                "VtDecoder: frame #{} produced no decodable NAL units after filtering",
                self.frame_count
            );
            return Ok(());
        }

        // Create CMBlockBuffer.
        // IMPORTANT: Pass kCFAllocatorNull as blockAllocator so CoreMedia does NOT
        // try to free our Rust-owned memory. We keep `data` alive until after decode.
        let mut block_buffer: CMBlockBufferRef = ptr::null();
        let status = unsafe {
            CMBlockBufferCreateWithMemoryBlock(
                kCFAllocatorDefault, // structureAllocator (for the CMBlockBuffer object itself)
                data.as_ptr() as *const c_void,
                data.len(),
                kCFAllocatorNull, // blockAllocator = NULL → CoreMedia won't free the memory
                ptr::null(),
                0,
                data.len(),
                0,
                &mut block_buffer,
            )
        };
        if status != 0 || block_buffer.is_null() {
            bail!("CMBlockBufferCreateWithMemoryBlock failed: {status}");
        }

        // Create CMSampleBuffer
        let sample_size = data.len();
        let pts = {
            let pts_duration: Duration = packet.timestamp.into();
            cm_time(pts_duration.as_micros() as i64, 1_000_000)
        };
        let timing = CMSampleTimingInfo {
            duration: K_CM_TIME_INVALID,
            presentation_time_stamp: pts,
            decode_time_stamp: K_CM_TIME_INVALID,
        };
        let mut sample_buffer: CMSampleBufferRef = ptr::null();
        let status = unsafe {
            CMSampleBufferCreateReady(
                kCFAllocatorDefault,
                block_buffer,
                self.format_desc,
                1, // num_samples
                1, // num_sample_timing_entries
                &timing,
                1, // num_sample_size_entries
                &sample_size,
                &mut sample_buffer,
            )
        };

        unsafe { CFRelease(block_buffer as _) };

        if status != 0 || sample_buffer.is_null() {
            bail!("CMSampleBufferCreate failed: {status}");
        }

        // Decode (synchronous — data must stay alive through this call)
        let mut info_flags: u32 = 0;
        let dec_status = unsafe {
            VTDecompressionSessionDecodeFrame(
                self.session,
                sample_buffer,
                0, // synchronous decode
                ptr::null_mut(),
                &mut info_flags,
            )
        };

        unsafe { CFRelease(sample_buffer as _) };

        // Now it's safe to drop data — decode is complete (synchronous)
        drop(data);

        if dec_status != 0 {
            tracing::warn!(
                "VTDecompressionSessionDecodeFrame failed: OSStatus={dec_status}, frame #{}",
                self.frame_count
            );
            // Don't bail — might be a recoverable error on a single frame
        } else if self.frame_count <= 10 || self.frame_count % 100 == 0 {
            // Check if the synchronous callback queued a frame
            let queue_len = self.output.lock().map(|q| q.len()).unwrap_or(999);
            tracing::info!(
                "VtDecoder: frame #{} decoded OK, output queue={}",
                self.frame_count,
                queue_len
            );
        }

        Ok(())
    }

    fn pop_frame(&mut self) -> Result<Option<DecodedFrame>> {
        let raw = {
            let mut q = self
                .output
                .lock()
                .map_err(|e| anyhow::anyhow!("lock: {e}"))?;
            let item = q.pop_front();
            if item.is_some() {
                let remaining = q.len();
                if remaining == 0 || remaining % 100 == 0 {
                    tracing::info!(
                        "VtDecoder::pop_frame: got frame, {} remaining in queue",
                        remaining
                    );
                }
            }
            item
        };

        match raw {
            None => Ok(None),
            Some(raw) => {
                // Create image::RgbaImage from RGBA bytes
                let img = image::RgbaImage::from_raw(raw.width, raw.height, raw.rgba_data)
                    .context("failed to create RgbaImage from decoded frame")?;
                let frame = image::Frame::new(img);
                let timestamp = Duration::from_micros(raw.timestamp_us);
                Ok(Some(DecodedFrame { frame, timestamp }))
            }
        }
    }

    fn set_viewport(&mut self, w: u32, h: u32) {
        self.viewport_w = w;
        self.viewport_h = h;
        // VTDecompressionSession doesn't support dynamic viewport changes;
        // the caller would need to rescale the output frame if needed.
    }
}

// ── VtDecoder private helpers ──

impl VtDecoder {
    /// Create an uninitialized decoder that will be initialized on the first keyframe.
    fn uninit(viewport_w: u32, viewport_h: u32) -> Self {
        Self {
            session: ptr::null_mut(),
            format_desc: ptr::null(),
            output: Arc::new(Mutex::new(VecDeque::new())),
            viewport_w,
            viewport_h,
            frame_count: 0,
            initialized: false,
        }
    }

    /// Create a VTDecompressionSession from avcC description bytes.
    ///
    /// Returns `(session, format_desc, output_queue)` on success.
    fn create_session(
        desc_bytes: &[u8],
    ) -> Result<(
        VTDecompressionSessionRef,
        CMFormatDescriptionRef,
        Arc<Mutex<VecDeque<RawDecodedFrame>>>,
    )> {
        // Parse avcC to extract SPS and PPS
        let (sps, pps) = parse_avcc(desc_bytes)?;

        // Create CMFormatDescription from SPS + PPS
        let param_sets: [*const u8; 2] = [sps.as_ptr(), pps.as_ptr()];
        let param_sizes: [usize; 2] = [sps.len(), pps.len()];
        let mut format_desc: CMFormatDescriptionRef = ptr::null();
        let status = unsafe {
            CMVideoFormatDescriptionCreateFromH264ParameterSets(
                kCFAllocatorDefault,
                2,
                param_sets.as_ptr(),
                param_sizes.as_ptr(),
                4, // 4-byte length-prefixed NALUs
                &mut format_desc,
            )
        };
        if status != 0 || format_desc.is_null() {
            bail!("CMVideoFormatDescriptionCreateFromH264ParameterSets failed: {status}");
        }

        // Destination pixel buffer attributes — request BGRA output
        let dest_attrs = unsafe {
            let dict = CFDictionaryCreateMutable(
                kCFAllocatorDefault,
                1,
                &kCFTypeDictionaryKeyCallBacks as *const _ as *const c_void,
                &kCFTypeDictionaryValueCallBacks as *const _ as *const c_void,
            );
            let pixel_format_val: i32 = K_CV_PIXEL_FORMAT_TYPE_32_BGRA as i32;
            let cf_num = CFNumberCreate(
                kCFAllocatorDefault,
                K_CF_NUMBER_SINT32_TYPE,
                &pixel_format_val as *const i32 as *const c_void,
            );
            CFDictionarySetValue(
                dict,
                kCVPixelBufferPixelFormatTypeKey as *const c_void,
                cf_num as *const c_void,
            );
            CFRelease(cf_num as CFTypeRef);
            dict
        };

        let output = Arc::new(Mutex::new(VecDeque::<RawDecodedFrame>::new()));
        let output_ptr = Arc::into_raw(output.clone()) as *mut c_void;

        let callback = VTDecompressionOutputCallbackRecord {
            decompress_output_callback: vt_decompress_callback,
            decompress_output_ref_con: output_ptr,
        };

        let mut session: VTDecompressionSessionRef = ptr::null_mut();
        let status = unsafe {
            VTDecompressionSessionCreate(
                kCFAllocatorDefault,
                format_desc,
                ptr::null(), // videoDecoderSpecification
                dest_attrs as CFDictionaryRef,
                &callback,
                &mut session,
            )
        };

        unsafe { CFRelease(dest_attrs as CFTypeRef) };

        if status != 0 || session.is_null() {
            // Clean up format_desc on failure
            unsafe { CFRelease(format_desc as CFTypeRef) };
            bail!("VTDecompressionSessionCreate failed: {status}");
        }

        tracing::info!("VtDecoder::create_session: success, session={session:?}");
        Ok((session, format_desc, output))
    }

    /// Extract SPS and PPS from AVCC-formatted (length-prefixed) keyframe data.
    ///
    /// Walks 4-byte length-prefixed NALUs and looks for SPS (type 7) and PPS (type 8).
    fn try_extract_sps_pps_from_avcc(&self, data: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
        let mut sps: Option<Vec<u8>> = None;
        let mut pps: Option<Vec<u8>> = None;
        let mut offset = 0;

        while offset + 4 <= data.len() {
            let nalu_len = u32::from_be_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;

            if nalu_len == 0 || offset + nalu_len > data.len() {
                break;
            }

            let nal_type = data[offset] & 0x1F;
            match nal_type {
                7 => {
                    tracing::info!(
                        "try_extract_sps_pps_from_avcc: found SPS ({} bytes)",
                        nalu_len
                    );
                    sps = Some(data[offset..offset + nalu_len].to_vec());
                }
                8 => {
                    tracing::info!(
                        "try_extract_sps_pps_from_avcc: found PPS ({} bytes)",
                        nalu_len
                    );
                    pps = Some(data[offset..offset + nalu_len].to_vec());
                }
                _ => {}
            }

            offset += nalu_len;
        }

        match (sps, pps) {
            (Some(s), Some(p)) => Ok((s, p)),
            (None, _) => bail!("AVCC keyframe: no SPS NAL unit found"),
            (_, None) => bail!("AVCC keyframe: no PPS NAL unit found"),
        }
    }
}

/// Check if data starts with an Annex B start code (00 00 00 01 or 00 00 01).
///
/// The 4-byte check is unambiguous (length=1 NAL units don't occur in practice).
/// The 3-byte check needs validation: AVCC data with NAL lengths 256-511 would
/// have bytes [0x00, 0x00, 0x01, ...] which looks like a 3-byte start code.
/// We validate by checking the NAL header byte (forbidden_zero_bit=0, type=1-23).
fn is_annex_b(data: &[u8]) -> bool {
    // 4-byte start code: unambiguous
    if data.len() >= 4 && data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1 {
        return true;
    }
    // 3-byte start code: validate the NAL header to avoid AVCC false positives
    if data.len() >= 4 && data[0] == 0 && data[1] == 0 && data[2] == 1 {
        let nal_byte = data[3];
        let forbidden_bit = nal_byte >> 7;
        let nal_type = nal_byte & 0x1F;
        // Valid H.264 NAL: forbidden_zero_bit=0, type 1-23
        return forbidden_bit == 0 && nal_type >= 1 && nal_type <= 23;
    }
    false
}

/// Convert Annex B byte stream (00 00 00 01 delimited NALUs) to
/// length-prefixed format (4-byte big-endian length + NALU data).
///
/// Filters out SEI NAL units (type 6) which VideoToolbox often rejects,
/// and AUD NAL units (type 9) which are unnecessary for decoding.
fn annex_b_to_length_prefixed(data: &[u8]) -> Vec<u8> {
    let nalu_starts = find_annex_b_nalu_starts(data);
    let mut out = Vec::with_capacity(data.len());

    for (idx, &start) in nalu_starts.iter().enumerate() {
        if start >= data.len() {
            continue;
        }
        let end = if idx + 1 < nalu_starts.len() {
            let next = nalu_starts[idx + 1];
            if next >= 4
                && data[next - 4] == 0
                && data[next - 3] == 0
                && data[next - 2] == 0
                && data[next - 1] == 1
            {
                next - 4
            } else if next >= 3 && data[next - 3] == 0 && data[next - 2] == 0 && data[next - 1] == 1
            {
                next - 3
            } else {
                next
            }
        } else {
            data.len()
        };

        let nalu = &data[start..end];
        if nalu.is_empty() {
            continue;
        }

        let nal_type = nalu[0] & 0x1F;
        // Skip SEI (6) and AUD (9) — VT doesn't need them and they can cause errors
        if nal_type == 6 || nal_type == 9 {
            continue;
        }
        // Skip SPS (7) and PPS (8) inline — already in the format description
        if nal_type == 7 || nal_type == 8 {
            continue;
        }

        let len = nalu.len() as u32;
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(nalu);
    }

    out
}

/// Find all NAL unit start positions (byte after the start code) in Annex B data.
fn find_annex_b_nalu_starts(data: &[u8]) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i < data.len() {
        if i + 3 < data.len()
            && data[i] == 0
            && data[i + 1] == 0
            && data[i + 2] == 0
            && data[i + 3] == 1
        {
            starts.push(i + 4);
            i += 4;
        } else if i + 2 < data.len() && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    starts
}

/// Parse H.264 codec description into SPS and PPS byte slices.
///
/// Supports two formats:
/// 1. **avcC box** (ISO 14496-15): version 0 or 1, followed by profile/compat/level/length_size,
///    then SPS count + SPS data + PPS count + PPS data.
/// 2. **Annex B**: start codes (`00 00 00 01` or `00 00 01`) separating NAL units.
///    SPS is NAL type 7, PPS is NAL type 8.
fn parse_avcc(data: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
    if data.len() < 4 {
        bail!("H.264 description too short: {} bytes", data.len());
    }

    // Log first bytes for debugging
    let hex_preview: String = data
        .iter()
        .take(32)
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ");
    tracing::info!("parse_avcc: {} bytes, first 32: {hex_preview}", data.len());

    // Detect Annex B format: starts with 00 00 00 01 or 00 00 01
    if (data.len() >= 4 && data[..4] == [0, 0, 0, 1]) || (data.len() >= 3 && data[..3] == [0, 0, 1])
    {
        tracing::info!("parse_avcc: detected Annex B format, parsing NAL units");
        return parse_annex_b(data);
    }

    // avcC box format
    if data[0] > 1 {
        bail!("avcC version {} not supported (expected 0 or 1)", data[0]);
    }

    if data.len() < 8 {
        bail!("avcC too short for box format: {} bytes", data.len());
    }

    let mut offset = 5; // skip version(1) + profile(1) + compat(1) + level(1) + length_size(1)

    // SPS count (lower 5 bits)
    let num_sps = (data[offset] & 0x1F) as usize;
    offset += 1;

    if num_sps == 0 {
        bail!("avcC: no SPS (num_sps=0)");
    }

    // Read first SPS
    if offset + 2 > data.len() {
        bail!(
            "avcC: truncated SPS length at offset {offset}, data len {}",
            data.len()
        );
    }
    let sps_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
    offset += 2;
    tracing::info!(
        "parse_avcc: num_sps={num_sps}, sps_len={sps_len}, offset={offset}, remaining={}",
        data.len() - offset
    );
    if offset + sps_len > data.len() {
        bail!(
            "avcC: truncated SPS data (need {sps_len} at offset {offset}, have {})",
            data.len() - offset
        );
    }
    let sps = data[offset..offset + sps_len].to_vec();
    offset += sps_len;

    // Skip remaining SPS if any
    for _ in 1..num_sps {
        if offset + 2 > data.len() {
            break;
        }
        let len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2 + len;
    }

    // PPS count
    if offset >= data.len() {
        bail!("avcC: no PPS section");
    }
    let num_pps = data[offset] as usize;
    offset += 1;

    if num_pps == 0 {
        bail!("avcC: no PPS");
    }

    // Read first PPS
    if offset + 2 > data.len() {
        bail!("avcC: truncated PPS length");
    }
    let pps_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
    offset += 2;
    if offset + pps_len > data.len() {
        bail!("avcC: truncated PPS data");
    }
    let pps = data[offset..offset + pps_len].to_vec();

    tracing::info!(
        "parse_avcc: OK, sps={} bytes, pps={} bytes",
        sps.len(),
        pps.len()
    );
    Ok((sps, pps))
}

/// Parse Annex B byte stream to extract SPS (NAL type 7) and PPS (NAL type 8).
fn parse_annex_b(data: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut sps: Option<Vec<u8>> = None;
    let mut pps: Option<Vec<u8>> = None;

    // Find all NAL unit boundaries (00 00 00 01 or 00 00 01)
    let mut i = 0;
    let mut nalu_starts: Vec<usize> = Vec::new();
    while i < data.len() {
        if i + 3 < data.len()
            && data[i] == 0
            && data[i + 1] == 0
            && data[i + 2] == 0
            && data[i + 3] == 1
        {
            nalu_starts.push(i + 4);
            i += 4;
        } else if i + 2 < data.len() && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            nalu_starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }

    for (idx, &start) in nalu_starts.iter().enumerate() {
        if start >= data.len() {
            continue;
        }
        let end = if idx + 1 < nalu_starts.len() {
            // Find the start code before the next NAL unit
            let next = nalu_starts[idx + 1];
            // Back up past the start code
            if next >= 4
                && data[next - 4] == 0
                && data[next - 3] == 0
                && data[next - 2] == 0
                && data[next - 1] == 1
            {
                next - 4
            } else if next >= 3 && data[next - 3] == 0 && data[next - 2] == 0 && data[next - 1] == 1
            {
                next - 3
            } else {
                next
            }
        } else {
            data.len()
        };

        let nal_data = &data[start..end];
        if nal_data.is_empty() {
            continue;
        }

        let nal_type = nal_data[0] & 0x1F;
        match nal_type {
            7 => {
                tracing::info!("parse_annex_b: found SPS ({} bytes)", nal_data.len());
                sps = Some(nal_data.to_vec());
            }
            8 => {
                tracing::info!("parse_annex_b: found PPS ({} bytes)", nal_data.len());
                pps = Some(nal_data.to_vec());
            }
            _ => {}
        }
    }

    match (sps, pps) {
        (Some(s), Some(p)) => Ok((s, p)),
        (None, _) => bail!("Annex B: no SPS NAL unit found"),
        (_, None) => bail!("Annex B: no PPS NAL unit found"),
    }
}

/// VTDecompressionSession output callback.
///
/// # Safety
/// `decompress_output_ref_con` must be a valid `Arc<Mutex<VecDeque<RawDecodedFrame>>>` raw pointer.
unsafe extern "C" fn vt_decompress_callback(
    decompress_output_ref_con: *mut c_void,
    _source_frame_ref_con: *mut c_void,
    status: OSStatus,
    _info_flags: u32,
    image_buffer: CVImageBufferRef,
    presentation_timestamp: CMTimeRepr,
    _presentation_duration: CMTimeRepr,
) {
    if status != 0 || image_buffer.is_null() {
        tracing::warn!(
            "VT decompress callback: status={status}, buffer null={}",
            image_buffer.is_null()
        );
        return;
    }

    let output = &*(decompress_output_ref_con as *const Mutex<VecDeque<RawDecodedFrame>>);

    // Lock pixel buffer to read BGRA data
    let lock_status = CVPixelBufferLockBaseAddress(image_buffer, 1); // 1 = read-only
    if lock_status != 0 {
        tracing::warn!("CVPixelBufferLockBaseAddress failed: {lock_status}");
        return;
    }

    let base = CVPixelBufferGetBaseAddress(image_buffer);
    let bytes_per_row = CVPixelBufferGetBytesPerRow(image_buffer);
    let width = CVPixelBufferGetWidth(image_buffer) as u32;
    let height = CVPixelBufferGetHeight(image_buffer) as u32;

    if !base.is_null() && width > 0 && height > 0 {
        // Convert BGRA → RGBA in-place while copying
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            let row_ptr = (base as *const u8).add(y as usize * bytes_per_row);
            for x in 0..width {
                let px = row_ptr.add(x as usize * 4);
                let b = *px;
                let g = *px.add(1);
                let r = *px.add(2);
                let a = *px.add(3);
                rgba.push(r);
                rgba.push(g);
                rgba.push(b);
                rgba.push(a);
            }
        }

        let timestamp_us = if presentation_timestamp.flags & K_CM_TIME_FLAGS_VALID != 0
            && presentation_timestamp.timescale > 0
        {
            (presentation_timestamp.value as f64 / presentation_timestamp.timescale as f64
                * 1_000_000.0) as u64
        } else {
            0
        };

        let frame = RawDecodedFrame {
            rgba_data: rgba,
            width,
            height,
            timestamp_us,
        };

        if let Ok(mut q) = output.lock() {
            let qlen = q.len();
            q.push_back(frame);
            // Log first 3 decoded frames and every 50th queued frame
            if qlen == 0 || qlen == 1 || qlen == 2 || (qlen + 1) % 50 == 0 {
                tracing::info!(
                    "VT callback: decoded frame queued, {width}x{height}, queue_len={}, ts={timestamp_us}us",
                    qlen + 1
                );
            }
        }
    } else {
        tracing::warn!(
            "VT callback: null/zero-size buffer, base_null={}, {width}x{height}",
            base.is_null()
        );
    }

    CVPixelBufferUnlockBaseAddress(image_buffer, 1);
}
