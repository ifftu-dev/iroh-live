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


//! VideoToolbox H.264 encoder for iOS.
//!
//! Uses `VTCompressionSession` to encode BGRA `VideoFrame`s into H.264
//! with length-prefixed NAL units (not Annex-B start codes).
//!
//! The encoder extracts avcC extradata (SPS+PPS) from the first
//! `CMFormatDescription` and stores it in the `VideoConfig.description` field,
//! which the iroh-live protocol requires for decoder initialization on the
//! receiving side.
//!
//! ## Safety
//!
//! This module uses `unsafe` extensively to call C APIs from CoreMedia,
//! CoreVideo, and VideoToolbox. All pointers are validated before use and
//! all CF objects are released via `CFRelease`.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use buf_list::BufList;
use bytes::Bytes;
use hang::Timestamp;

use crate::av::{VideoEncoder, VideoEncoderInner, VideoFrame, VideoPreset};

// ── CoreFoundation / CoreMedia / CoreVideo / VideoToolbox C types ──

type CFAllocatorRef = *const c_void;
type CFDictionaryRef = *const c_void;
type CFMutableDictionaryRef = *mut c_void;
type CFStringRef = *const c_void;
type CFNumberRef = *const c_void;
type CFTypeRef = *const c_void;
type CFBooleanRef = *const c_void;
type CMTime = CMTimeRepr;
type CMSampleBufferRef = *const c_void;
type CMFormatDescriptionRef = *const c_void;
type CMBlockBufferRef = *const c_void;
type CVPixelBufferRef = *const c_void;
type CVPixelBufferPoolRef = *const c_void;
type VTCompressionSessionRef = *mut c_void;
type OSStatus = i32;

#[repr(C)]
#[derive(Copy, Clone)]
struct CMTimeRepr {
    value: i64,
    timescale: i32,
    flags: u32,
    epoch: i64,
}

// CMTime flags
const K_CM_TIME_FLAGS_VALID: u32 = 1;

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

// Pixel format: kCVPixelFormatType_32BGRA
const K_CV_PIXEL_FORMAT_TYPE_32_BGRA: u32 = 0x42475241; // 'BGRA'

// CFNumber types
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
    static kCFBooleanTrue: CFBooleanRef;
    static kCFBooleanFalse: CFBooleanRef;

    // CoreVideo
    static kCVPixelBufferPixelFormatTypeKey: CFStringRef;
    static kCVPixelBufferWidthKey: CFStringRef;
    static kCVPixelBufferHeightKey: CFStringRef;
    static kCVPixelBufferIOSurfacePropertiesKey: CFStringRef;

    fn CVPixelBufferPoolCreatePixelBuffer(
        allocator: CFAllocatorRef,
        pool: CVPixelBufferPoolRef,
        pixel_buffer_out: *mut CVPixelBufferRef,
    ) -> OSStatus;
    fn CVPixelBufferCreateWithBytes(
        allocator: CFAllocatorRef,
        width: usize,
        height: usize,
        pixel_format: u32,
        base_address: *mut c_void,
        bytes_per_row: usize,
        release_callback: *const c_void,
        release_ref_con: *mut c_void,
        pixel_buffer_attributes: CFDictionaryRef,
        pixel_buffer_out: *mut CVPixelBufferRef,
    ) -> OSStatus;
    fn CVPixelBufferLockBaseAddress(pixel_buffer: CVPixelBufferRef, flags: u64) -> OSStatus;
    fn CVPixelBufferUnlockBaseAddress(pixel_buffer: CVPixelBufferRef, flags: u64) -> OSStatus;
    fn CVPixelBufferGetBaseAddress(pixel_buffer: CVPixelBufferRef) -> *const c_void;
    fn CVPixelBufferGetBytesPerRow(pixel_buffer: CVPixelBufferRef) -> usize;

    // CoreMedia
    fn CMSampleBufferGetFormatDescription(sbuf: CMSampleBufferRef) -> CMFormatDescriptionRef;
    fn CMSampleBufferGetDataBuffer(sbuf: CMSampleBufferRef) -> CMBlockBufferRef;
    fn CMSampleBufferGetPresentationTimeStamp(sbuf: CMSampleBufferRef) -> CMTimeRepr;
    fn CMBlockBufferGetDataLength(block: CMBlockBufferRef) -> usize;
    fn CMBlockBufferCopyDataBytes(
        block: CMBlockBufferRef,
        offset: usize,
        length: usize,
        dest: *mut c_void,
    ) -> OSStatus;
    fn CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
        format_desc: CMFormatDescriptionRef,
        index: usize,
        out_ptr: *mut *const u8,
        out_size: *mut usize,
        out_count: *mut usize,
        out_nal_unit_header_length: *mut i32,
    ) -> OSStatus;
    fn CMSampleBufferGetSampleAttachmentsArray(
        sbuf: CMSampleBufferRef,
        create_if_necessary: u8,
    ) -> *const c_void; // CFArrayRef
    fn CFArrayGetValueAtIndex(array: *const c_void, idx: isize) -> *const c_void; // CFDictionaryRef
    fn CFDictionaryGetValueIfPresent(
        dict: CFDictionaryRef,
        key: *const c_void,
        value_out: *mut *const c_void,
    ) -> u8;

    // kCMSampleAttachmentKey_NotSync
    static kCMSampleAttachmentKey_NotSync: CFStringRef;

    // VideoToolbox
    fn VTCompressionSessionCreate(
        allocator: CFAllocatorRef,
        width: i32,
        height: i32,
        codec_type: u32,
        encoder_specification: CFDictionaryRef,
        source_image_buffer_attributes: CFDictionaryRef,
        compressed_data_allocator: CFAllocatorRef,
        output_callback: *const c_void,
        output_callback_ref_con: *mut c_void,
        compression_session_out: *mut VTCompressionSessionRef,
    ) -> OSStatus;
    fn VTCompressionSessionPrepareToEncodeFrames(session: VTCompressionSessionRef) -> OSStatus;
    fn VTCompressionSessionEncodeFrame(
        session: VTCompressionSessionRef,
        image_buffer: CVPixelBufferRef,
        presentation_timestamp: CMTime,
        duration: CMTime,
        frame_properties: CFDictionaryRef,
        source_frame_ref_con: *mut c_void,
        info_flags_out: *mut u32,
    ) -> OSStatus;
    fn VTCompressionSessionCompleteFrames(
        session: VTCompressionSessionRef,
        complete_until_presentation_timestamp: CMTime,
    ) -> OSStatus;
    fn VTCompressionSessionInvalidate(session: VTCompressionSessionRef);
    fn VTCompressionSessionGetPixelBufferPool(
        session: VTCompressionSessionRef,
    ) -> CVPixelBufferPoolRef;
    fn VTSessionSetProperty(
        session: *mut c_void,
        property_key: CFStringRef,
        property_value: CFTypeRef,
    ) -> OSStatus;

    // VT property keys
    static kVTCompressionPropertyKey_RealTime: CFStringRef;
    static kVTCompressionPropertyKey_ProfileLevel: CFStringRef;
    static kVTCompressionPropertyKey_AllowFrameReordering: CFStringRef;
    static kVTCompressionPropertyKey_MaxKeyFrameInterval: CFStringRef;
    static kVTCompressionPropertyKey_ExpectedFrameRate: CFStringRef;
    static kVTCompressionPropertyKey_AverageBitRate: CFStringRef;

    // Profile level values
    static kVTProfileLevel_H264_Baseline_AutoLevel: CFStringRef;
}

// kCMVideoCodecType_H264
const K_CM_VIDEO_CODEC_TYPE_H264: u32 = 0x61766331; // 'avc1'

/// Shared output buffer between the VT callback and the encoder.
struct OutputQueue {
    packets: VecDeque<EncodedPacket>,
    extradata: Option<Bytes>,
}

struct EncodedPacket {
    data: Bytes,
    timestamp: Timestamp,
    keyframe: bool,
}

/// VideoToolbox H.264 encoder for iOS.
///
/// Encodes raw BGRA `VideoFrame`s into H.264 packets using Apple's
/// hardware-accelerated `VTCompressionSession`.
pub struct VtEncoder {
    session: VTCompressionSessionRef,
    output: Arc<Mutex<OutputQueue>>,
    width: u32,
    height: u32,
    fps: f64,
    bitrate: u32,
    frame_index: u64,
    config: hang::catalog::VideoConfig,
}

// SAFETY: The VTCompressionSession handle is thread-safe for the operations
// we perform (encode, complete, invalidate). The output queue is Mutex-guarded.
unsafe impl Send for VtEncoder {}

impl Drop for VtEncoder {
    fn drop(&mut self) {
        if !self.session.is_null() {
            unsafe {
                VTCompressionSessionInvalidate(self.session);
                CFRelease(self.session as CFTypeRef);
            }
            self.session = ptr::null_mut();
        }
    }
}

impl VtEncoder {
    fn create(width: u32, height: u32, fps: f64, bitrate: u32) -> Result<Self> {
        let output = Arc::new(Mutex::new(OutputQueue {
            packets: VecDeque::new(),
            extradata: None,
        }));

        let mut session: VTCompressionSessionRef = ptr::null_mut();

        // Create source pixel buffer attributes (BGRA)
        let pb_attrs = unsafe {
            let dict = CFDictionaryCreateMutable(
                kCFAllocatorDefault,
                4,
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

            let w = width as i32;
            let w_num = CFNumberCreate(
                kCFAllocatorDefault,
                K_CF_NUMBER_SINT32_TYPE,
                &w as *const _ as *const c_void,
            );
            CFDictionarySetValue(dict, kCVPixelBufferWidthKey as _, w_num as _);
            CFRelease(w_num as _);

            let h = height as i32;
            let h_num = CFNumberCreate(
                kCFAllocatorDefault,
                K_CF_NUMBER_SINT32_TYPE,
                &h as *const _ as *const c_void,
            );
            CFDictionarySetValue(dict, kCVPixelBufferHeightKey as _, h_num as _);
            CFRelease(h_num as _);

            // IOSurface properties (empty dict — enables IOSurface backing)
            let io_dict = CFDictionaryCreateMutable(
                kCFAllocatorDefault,
                0,
                &kCFTypeDictionaryKeyCallBacks as *const _ as *const c_void,
                &kCFTypeDictionaryValueCallBacks as *const _ as *const c_void,
            );
            CFDictionarySetValue(
                dict,
                kCVPixelBufferIOSurfacePropertiesKey as _,
                io_dict as _,
            );
            CFRelease(io_dict as _);

            dict
        };

        // Callback context: Arc<Mutex<OutputQueue>>
        let output_ptr = Arc::into_raw(output.clone()) as *mut c_void;

        let status = unsafe {
            VTCompressionSessionCreate(
                kCFAllocatorDefault,
                width as i32,
                height as i32,
                K_CM_VIDEO_CODEC_TYPE_H264,
                ptr::null(),         // encoder specification (let system choose)
                pb_attrs as _,       // source pixel buffer attributes
                kCFAllocatorDefault, // compressed data allocator
                vt_output_callback as *const c_void,
                output_ptr,
                &mut session,
            )
        };

        // Release pixel buffer attributes dictionary
        unsafe { CFRelease(pb_attrs as _) };

        if status != 0 {
            // Re-take the Arc so it doesn't leak
            unsafe { Arc::from_raw(output_ptr as *const Mutex<OutputQueue>) };
            bail!("VTCompressionSessionCreate failed: OSStatus {status}");
        }

        // Configure session properties
        unsafe {
            // Real-time encoding
            VTSessionSetProperty(
                session,
                kVTCompressionPropertyKey_RealTime,
                kCFBooleanTrue as _,
            );

            // H.264 Baseline profile (maximum compatibility)
            VTSessionSetProperty(
                session,
                kVTCompressionPropertyKey_ProfileLevel,
                kVTProfileLevel_H264_Baseline_AutoLevel as _,
            );

            // No B-frames (baseline doesn't support them, but be explicit)
            VTSessionSetProperty(
                session,
                kVTCompressionPropertyKey_AllowFrameReordering,
                kCFBooleanFalse as _,
            );

            // Keyframe interval: every 2 seconds
            let kf_interval = (fps * 2.0) as i32;
            let kf_num = CFNumberCreate(
                kCFAllocatorDefault,
                K_CF_NUMBER_SINT32_TYPE,
                &kf_interval as *const _ as *const c_void,
            );
            VTSessionSetProperty(
                session,
                kVTCompressionPropertyKey_MaxKeyFrameInterval,
                kf_num as _,
            );
            CFRelease(kf_num as _);

            // Expected frame rate
            let fps_i32 = fps as i32;
            let fps_num = CFNumberCreate(
                kCFAllocatorDefault,
                K_CF_NUMBER_SINT32_TYPE,
                &fps_i32 as *const _ as *const c_void,
            );
            VTSessionSetProperty(
                session,
                kVTCompressionPropertyKey_ExpectedFrameRate,
                fps_num as _,
            );
            CFRelease(fps_num as _);

            // Average bitrate
            let br = bitrate as i32;
            let br_num = CFNumberCreate(
                kCFAllocatorDefault,
                K_CF_NUMBER_SINT32_TYPE,
                &br as *const _ as *const c_void,
            );
            VTSessionSetProperty(
                session,
                kVTCompressionPropertyKey_AverageBitRate,
                br_num as _,
            );
            CFRelease(br_num as _);

            // Prepare to encode
            let status = VTCompressionSessionPrepareToEncodeFrames(session);
            if status != 0 {
                VTCompressionSessionInvalidate(session);
                CFRelease(session as _);
                Arc::from_raw(output_ptr as *const Mutex<OutputQueue>);
                bail!("VTCompressionSessionPrepareToEncodeFrames failed: OSStatus {status}");
            }
        }

        // Build initial VideoConfig (description will be filled when first keyframe arrives)
        let config = hang::catalog::VideoConfig {
            codec: hang::catalog::VideoCodec::H264(hang::catalog::H264 {
                profile: 0x42, // Baseline
                constraints: 0xE0,
                level: 0x1F, // Level 3.1
                inline: false,
            }),
            description: None,
            coded_width: Some(width),
            coded_height: Some(height),
            display_ratio_width: None,
            display_ratio_height: None,
            framerate: Some(fps),
            bitrate: Some(bitrate as u64),
            optimize_for_latency: Some(true),
        };

        Ok(Self {
            session,
            output,
            width,
            height,
            fps,
            bitrate,
            frame_index: 0,
            config,
        })
    }
}

impl VideoEncoder for VtEncoder {
    fn with_preset(preset: VideoPreset) -> Result<Self> {
        let (w, h) = preset.dimensions();
        let fps = preset.fps() as f64;
        let bitrate = match preset {
            VideoPreset::P180 => 300_000,
            VideoPreset::P360 => 800_000,
            VideoPreset::P720 => 2_500_000,
            VideoPreset::P1080 => 5_000_000,
        };
        // VtEncoder only runs on iOS phones which use portrait camera
        // orientation (480×640).  Swap preset dimensions so the session
        // matches the portrait aspect ratio (e.g. P180 → 180×320).
        Self::create(h, w, fps, bitrate)
    }
}

impl VideoEncoderInner for VtEncoder {
    fn name(&self) -> &str {
        "vt-h264"
    }

    fn config(&self) -> hang::catalog::VideoConfig {
        // Re-read extradata if it's been captured
        let mut cfg = self.config.clone();
        if cfg.description.is_none() {
            if let Ok(q) = self.output.lock() {
                if let Some(ref ed) = q.extradata {
                    cfg.description = Some(ed.clone());
                }
            }
        }
        cfg
    }

    fn push_frame(&mut self, frame: VideoFrame) -> Result<()> {
        // VTCompressionSession auto-scales input to the session's configured size,
        // so CVPixelBuffer must use the frame's actual dimensions, not the encoder's.
        let frame_width = frame.format.dimensions[0] as usize;
        let frame_height = frame.format.dimensions[1] as usize;
        let bytes_per_row = frame_width * 4;
        let expected_len = bytes_per_row * frame_height;
        if frame.raw.len() < expected_len {
            bail!(
                "frame too small: {} bytes, expected {} ({}x{}x4)",
                frame.raw.len(),
                expected_len,
                frame_width,
                frame_height,
            );
        }

        let mut pixel_buffer: CVPixelBufferRef = ptr::null();

        // We need a mutable copy because CVPixelBufferCreateWithBytes wants *mut
        let mut raw_copy = frame.raw.to_vec();

        let status = unsafe {
            CVPixelBufferCreateWithBytes(
                kCFAllocatorDefault,
                frame_width,
                frame_height,
                K_CV_PIXEL_FORMAT_TYPE_32_BGRA,
                raw_copy.as_mut_ptr() as *mut c_void,
                bytes_per_row,
                ptr::null(), // no release callback — we own the data
                ptr::null_mut(),
                ptr::null(), // no attributes
                &mut pixel_buffer,
            )
        };

        if status != 0 || pixel_buffer.is_null() {
            bail!("CVPixelBufferCreateWithBytes failed: OSStatus {status}");
        }

        // Presentation timestamp in microseconds, using 1_000_000 timescale
        let pts = cm_time(self.frame_index as i64, (self.fps as i32).max(1) * 1000);
        let duration = K_CM_TIME_INVALID;

        let mut info_flags: u32 = 0;
        let enc_status = unsafe {
            VTCompressionSessionEncodeFrame(
                self.session,
                pixel_buffer,
                pts,
                duration,
                ptr::null(),     // no per-frame properties
                ptr::null_mut(), // no source frame ref con
                &mut info_flags,
            )
        };

        // Release the pixel buffer
        unsafe { CFRelease(pixel_buffer as _) };

        // Keep raw_copy alive until here
        drop(raw_copy);

        if enc_status != 0 {
            bail!("VTCompressionSessionEncodeFrame failed: OSStatus {enc_status}");
        }

        self.frame_index += 1;
        Ok(())
    }

    fn pop_packet(&mut self) -> Result<Option<hang::Frame>> {
        let mut q = self
            .output
            .lock()
            .map_err(|e| anyhow::anyhow!("lock: {e}"))?;

        // Update config with extradata if available
        if self.config.description.is_none() {
            if let Some(ref ed) = q.extradata {
                self.config.description = Some(ed.clone());
            }
        }

        let pkt = match q.packets.pop_front() {
            Some(p) => p,
            None => return Ok(None),
        };

        let mut payload = BufList::new();
        payload.push_chunk(pkt.data);

        Ok(Some(hang::Frame {
            payload,
            timestamp: pkt.timestamp,
            keyframe: pkt.keyframe,
        }))
    }
}

// ── VT output callback ──

/// Called by VideoToolbox on the encoder's internal thread when a frame is ready.
///
/// # Safety
/// `output_callback_ref_con` must be a valid `Arc<Mutex<OutputQueue>>` raw pointer.
unsafe extern "C" fn vt_output_callback(
    output_callback_ref_con: *mut c_void,
    _source_frame_ref_con: *mut c_void,
    status: OSStatus,
    _info_flags: u32,
    sample_buffer: CMSampleBufferRef,
) {
    if status != 0 || sample_buffer.is_null() {
        tracing::warn!(
            "VT output callback: status={status}, buffer null={}",
            sample_buffer.is_null()
        );
        return;
    }

    let output = &*(output_callback_ref_con as *const Mutex<OutputQueue>);

    // Check if keyframe
    let keyframe = is_keyframe(sample_buffer);

    // Extract extradata from format description on keyframes
    if keyframe {
        let fmt = CMSampleBufferGetFormatDescription(sample_buffer);
        if !fmt.is_null() {
            if let Some(extradata) = extract_avcc_extradata(fmt) {
                if let Ok(mut q) = output.lock() {
                    q.extradata = Some(extradata);
                }
            }
        }
    }

    // Extract encoded data
    let block = CMSampleBufferGetDataBuffer(sample_buffer);
    if block.is_null() {
        return;
    }

    let data_len = CMBlockBufferGetDataLength(block);
    if data_len == 0 {
        return;
    }

    let mut buf = vec![0u8; data_len];
    let copy_status =
        CMBlockBufferCopyDataBytes(block, 0, data_len, buf.as_mut_ptr() as *mut c_void);
    if copy_status != 0 {
        tracing::warn!("CMBlockBufferCopyDataBytes failed: {copy_status}");
        return;
    }

    // For keyframes, prepend SPS+PPS as AVCC NAL units so the bitstream
    // is self-contained. This allows decoders that missed the out-of-band
    // avcC description (catalog race) to initialize from inline parameter sets.
    if keyframe {
        let fmt = CMSampleBufferGetFormatDescription(sample_buffer);
        if !fmt.is_null() {
            if let Some(sps_pps_nals) = extract_sps_pps_as_avcc_nals(fmt) {
                let prefix_len = sps_pps_nals.len();
                let mut combined = Vec::with_capacity(prefix_len + buf.len());
                combined.extend_from_slice(&sps_pps_nals);
                combined.extend_from_slice(&buf);
                tracing::debug!(
                    "VT encoder: prepended {} bytes SPS+PPS to keyframe ({} -> {} bytes)",
                    prefix_len,
                    data_len,
                    combined.len()
                );
                buf = combined;
            }
        }
    }

    // Get presentation timestamp
    let pts = CMSampleBufferGetPresentationTimeStamp(sample_buffer);
    let timestamp = if pts.flags & K_CM_TIME_FLAGS_VALID != 0 && pts.timescale > 0 {
        let micros = (pts.value as f64 / pts.timescale as f64 * 1_000_000.0) as u64;
        Timestamp::from_micros(micros).unwrap_or(Timestamp::ZERO)
    } else {
        Timestamp::ZERO
    };

    let pkt = EncodedPacket {
        data: Bytes::from(buf),
        timestamp,
        keyframe,
    };

    if let Ok(mut q) = output.lock() {
        q.packets.push_back(pkt);
    }
}

/// Check if a sample buffer represents a keyframe.
unsafe fn is_keyframe(sbuf: CMSampleBufferRef) -> bool {
    let attachments = CMSampleBufferGetSampleAttachmentsArray(sbuf, 0);
    if attachments.is_null() {
        return true; // no attachments → assume keyframe
    }
    let dict = CFArrayGetValueAtIndex(attachments, 0);
    if dict.is_null() {
        return true;
    }
    let mut value: *const c_void = ptr::null();
    let has_not_sync =
        CFDictionaryGetValueIfPresent(dict, kCMSampleAttachmentKey_NotSync as _, &mut value);
    if has_not_sync != 0 && value == kCFBooleanTrue as *const c_void {
        false // NotSync = true → not a keyframe
    } else {
        true
    }
}

/// Extract SPS and PPS from a CMFormatDescription and return them as
/// AVCC-format NAL units (4-byte big-endian length prefix + NAL data).
///
/// This is prepended to keyframe packets so decoders can initialize from
/// inline parameter sets, even without the out-of-band avcC description
/// (which may be missing due to catalog race conditions).
unsafe fn extract_sps_pps_as_avcc_nals(fmt: CMFormatDescriptionRef) -> Option<Vec<u8>> {
    let mut sps_ptr: *const u8 = ptr::null();
    let mut sps_size: usize = 0;
    let mut param_count: usize = 0;
    let mut nal_header_len: i32 = 0;

    let status = CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
        fmt,
        0,
        &mut sps_ptr,
        &mut sps_size,
        &mut param_count,
        &mut nal_header_len,
    );
    if status != 0 || sps_ptr.is_null() || sps_size == 0 {
        return None;
    }
    let sps = std::slice::from_raw_parts(sps_ptr, sps_size);

    let mut pps_ptr: *const u8 = ptr::null();
    let mut pps_size: usize = 0;
    let status = CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
        fmt,
        1,
        &mut pps_ptr,
        &mut pps_size,
        &mut param_count,
        &mut nal_header_len,
    );
    if status != 0 || pps_ptr.is_null() || pps_size == 0 {
        return None;
    }
    let pps = std::slice::from_raw_parts(pps_ptr, pps_size);

    // Build AVCC NAL units: [4-byte big-endian length][NAL data] for each
    let mut out = Vec::with_capacity(8 + sps_size + pps_size);
    out.extend_from_slice(&(sps_size as u32).to_be_bytes());
    out.extend_from_slice(sps);
    out.extend_from_slice(&(pps_size as u32).to_be_bytes());
    out.extend_from_slice(pps);

    Some(out)
}

/// Extract avcC extradata (SPS + PPS) from a CMFormatDescription.
///
/// The avcC box format:
/// ```text
/// [1 byte] version = 1
/// [1 byte] profile (from SPS[1])
/// [1 byte] profile_compat (from SPS[2])
/// [1 byte] level (from SPS[3])
/// [1 byte] length_size_minus_one = 3 (4-byte length-prefixed NALUs)
/// [1 byte] num_sps | 0xE0
/// [2 bytes] sps_length (big-endian)
/// [sps_length bytes] sps_data
/// [1 byte] num_pps
/// [2 bytes] pps_length (big-endian)
/// [pps_length bytes] pps_data
/// ```
unsafe fn extract_avcc_extradata(fmt: CMFormatDescriptionRef) -> Option<Bytes> {
    let mut sps_ptr: *const u8 = ptr::null();
    let mut sps_size: usize = 0;
    let mut param_count: usize = 0;
    let mut nal_header_len: i32 = 0;

    // Get SPS (index 0)
    let status = CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
        fmt,
        0,
        &mut sps_ptr,
        &mut sps_size,
        &mut param_count,
        &mut nal_header_len,
    );
    if status != 0 || sps_ptr.is_null() || sps_size < 4 {
        return None;
    }

    let sps = std::slice::from_raw_parts(sps_ptr, sps_size);

    // Get PPS (index 1)
    let mut pps_ptr: *const u8 = ptr::null();
    let mut pps_size: usize = 0;
    let status = CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
        fmt,
        1,
        &mut pps_ptr,
        &mut pps_size,
        &mut param_count,
        &mut nal_header_len,
    );
    if status != 0 || pps_ptr.is_null() || pps_size == 0 {
        return None;
    }

    let pps = std::slice::from_raw_parts(pps_ptr, pps_size);

    // Build avcC box
    let mut avcc = Vec::with_capacity(11 + sps_size + pps_size);
    avcc.push(1); // version
    avcc.push(sps[1]); // profile
    avcc.push(sps[2]); // profile_compat
    avcc.push(sps[3]); // level
    avcc.push(0xFF); // length_size_minus_one = 3 (4-byte) | reserved 0xFC
    avcc.push(0xE1); // num_sps = 1 | reserved 0xE0
    avcc.push((sps_size >> 8) as u8);
    avcc.push(sps_size as u8);
    avcc.extend_from_slice(sps);
    avcc.push(1); // num_pps
    avcc.push((pps_size >> 8) as u8);
    avcc.push(pps_size as u8);
    avcc.extend_from_slice(pps);

    Some(Bytes::from(avcc))
}
