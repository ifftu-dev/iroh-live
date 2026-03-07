//! VideoToolbox H.264 decoder for iOS.
//!
//! Uses `VTDecompressionSession` to decode H.264 length-prefixed NAL units
//! back into RGBA frames (as `DecodedFrame` with `image::Frame`).

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{bail, Context, Result};
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
    fn CMSampleBufferCreate(
        allocator: CFAllocatorRef,
        data_buffer: CMBlockBufferRef,
        data_ready: u8,
        make_data_ready_callback: *const c_void,
        make_data_ready_ref_con: *mut c_void,
        format_description: CMFormatDescriptionRef,
        num_samples: i32,
        num_sample_timing_entries: i32,
        sample_timing_array: *const c_void,
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
pub struct VtDecoder {
    session: VTDecompressionSessionRef,
    format_desc: CMFormatDescriptionRef,
    output: Arc<Mutex<VecDeque<RawDecodedFrame>>>,
    viewport_w: u32,
    viewport_h: u32,
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
        let desc_bytes = config
            .description
            .as_ref()
            .context("VideoConfig.description (avcC) is required for H.264 decoding")?;

        // Parse avcC to extract SPS and PPS
        let (sps, pps) = parse_avcc(desc_bytes)?;

        // Create CMFormatDescription from SPS+PPS
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

        // Destination pixel buffer attributes: BGRA (we'll convert to RGBA in software)
        let dest_attrs = unsafe {
            let dict = CFDictionaryCreateMutable(
                kCFAllocatorDefault,
                1,
                &kCFTypeDictionaryKeyCallBacks as *const _ as *const c_void,
                &kCFTypeDictionaryValueCallBacks as *const _ as *const c_void,
            );
            let fmt = K_CV_PIXEL_FORMAT_TYPE_32_BGRA as i32;
            let fmt_num = CFNumberCreate(
                kCFAllocatorDefault,
                K_CF_NUMBER_SINT32_TYPE,
                &fmt as *const _ as *const c_void,
            );
            CFDictionarySetValue(dict, kCVPixelBufferPixelFormatTypeKey as _, fmt_num as _);
            CFRelease(fmt_num as _);
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
                ptr::null(),     // decoder specification
                dest_attrs as _, // destination attributes
                &callback,
                &mut session,
            )
        };

        unsafe { CFRelease(dest_attrs as _) };

        if status != 0 || session.is_null() {
            unsafe {
                CFRelease(format_desc as _);
                Arc::from_raw(output_ptr as *const Mutex<VecDeque<RawDecodedFrame>>);
            }
            bail!("VTDecompressionSessionCreate failed: {status}");
        }

        Ok(Self {
            session,
            format_desc,
            output,
            viewport_w: config.coded_width.unwrap_or(640),
            viewport_h: config.coded_height.unwrap_or(480),
        })
    }

    fn name(&self) -> &str {
        "vt-h264-dec"
    }

    fn push_packet(&mut self, packet: hang::Frame) -> Result<()> {
        // Collect payload bytes
        let data: Vec<u8> = packet
            .payload
            .iter()
            .flat_map(|b| b.iter().copied())
            .collect();
        if data.is_empty() {
            return Ok(());
        }

        // Create CMBlockBuffer
        let mut block_buffer: CMBlockBufferRef = ptr::null();
        let status = unsafe {
            CMBlockBufferCreateWithMemoryBlock(
                kCFAllocatorDefault,
                data.as_ptr() as *const c_void,
                data.len(),
                kCFAllocatorDefault,
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
        let mut sample_buffer: CMSampleBufferRef = ptr::null();
        let status = unsafe {
            CMSampleBufferCreate(
                kCFAllocatorDefault,
                block_buffer,
                1, // data_ready
                ptr::null(),
                ptr::null_mut(),
                self.format_desc,
                1, // num_samples
                0, // num_sample_timing_entries (no timing)
                ptr::null(),
                1, // num_sample_size_entries
                &sample_size,
                &mut sample_buffer,
            )
        };

        unsafe { CFRelease(block_buffer as _) };

        if status != 0 || sample_buffer.is_null() {
            bail!("CMSampleBufferCreate failed: {status}");
        }

        // Decode
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

        // Keep data alive for the block buffer's lifetime
        drop(data);

        if dec_status != 0 {
            tracing::warn!("VTDecompressionSessionDecodeFrame failed: {dec_status}");
            // Don't bail — might be a recoverable error on a single frame
        }

        Ok(())
    }

    fn pop_frame(&mut self) -> Result<Option<DecodedFrame>> {
        let raw = {
            let mut q = self
                .output
                .lock()
                .map_err(|e| anyhow::anyhow!("lock: {e}"))?;
            q.pop_front()
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
            q.push_back(frame);
        }
    }

    CVPixelBufferUnlockBaseAddress(image_buffer, 1);
}
