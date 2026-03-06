//! iOS camera capture using AVFoundation (AVCaptureSession).
//!
//! Implements the `VideoSource` trait by capturing BGRA frames from the
//! device camera via Objective-C runtime FFI. No `objc2` crate dependency —
//! uses raw `objc_msgSend` calls for maximum control and minimal deps.
//!
//! ## Architecture
//!
//! 1. Creates an `AVCaptureSession` with `.medium` preset (640x480 BGRA).
//! 2. Adds an `AVCaptureDeviceInput` for the back camera (or front as fallback).
//! 3. Adds an `AVCaptureVideoDataOutput` configured for BGRA pixel format.
//! 4. Sets a delegate (Rust-allocated ObjC class) that receives
//!    `captureOutput:didOutputSampleBuffer:fromConnection:` callbacks.
//! 5. The delegate callback locks the CVPixelBuffer, copies BGRA data, and
//!    pushes it into a `std::sync::mpsc::Sender<VideoFrame>`.
//! 6. `pop_frame()` drains the receiver, returning the latest frame.
//!
//! ## Safety
//!
//! Uses extensive `unsafe` for ObjC runtime and CoreVideo FFI.
//! All ObjC objects are retained/released correctly.

use std::ffi::c_void;
use std::ptr;
use std::sync::mpsc;

use anyhow::{bail, Result};
use bytes::Bytes;

use crate::av::{PixelFormat, VideoFormat, VideoFrame, VideoSource};

// ── ObjC runtime types ──────────────────────────────────────────────

type Id = *mut c_void;
type Class = *mut c_void;
type Sel = *mut c_void;
type IMP = *const c_void;
type BOOL = i8;

const NIL: Id = ptr::null_mut();
const YES: BOOL = 1;
#[allow(dead_code)]
const NO: BOOL = 0;

// CVPixelBuffer types (re-used from encoder)
type CVPixelBufferRef = *const c_void;
type CMSampleBufferRef = *const c_void;

const K_CV_PIXEL_FORMAT_TYPE_32_BGRA: u32 = 0x42475241;

unsafe extern "C" {
    // ObjC runtime
    fn objc_getClass(name: *const u8) -> Class;
    fn sel_registerName(name: *const u8) -> Sel;
    fn objc_msgSend(receiver: Id, sel: Sel, ...) -> Id;
    fn objc_allocateClassPair(superclass: Class, name: *const u8, extra_bytes: usize) -> Class;
    fn objc_registerClassPair(cls: Class);
    fn class_addMethod(cls: Class, sel: Sel, imp: IMP, types: *const u8) -> BOOL;
    fn class_addIvar(
        cls: Class,
        name: *const u8,
        size: usize,
        alignment: u8,
        types: *const u8,
    ) -> BOOL;
    fn object_getInstanceVariable(obj: Id, name: *const u8, out_value: *mut *mut c_void) -> Id;
    fn object_setInstanceVariable(obj: Id, name: *const u8, value: *mut c_void) -> Id;

    // CoreVideo
    fn CVPixelBufferLockBaseAddress(pb: CVPixelBufferRef, flags: u64) -> i32;
    fn CVPixelBufferUnlockBaseAddress(pb: CVPixelBufferRef, flags: u64) -> i32;
    fn CVPixelBufferGetBaseAddress(pb: CVPixelBufferRef) -> *const c_void;
    fn CVPixelBufferGetBytesPerRow(pb: CVPixelBufferRef) -> usize;
    fn CVPixelBufferGetWidth(pb: CVPixelBufferRef) -> usize;
    fn CVPixelBufferGetHeight(pb: CVPixelBufferRef) -> usize;

    // CoreMedia
    fn CMSampleBufferGetImageBuffer(sbuf: CMSampleBufferRef) -> CVPixelBufferRef;
}

// Helper macros for ObjC messaging
macro_rules! msg_send {
    ($obj:expr, $sel:expr) => {{
        let sel = sel_registerName(concat!($sel, "\0").as_ptr());
        objc_msgSend($obj, sel)
    }};
    ($obj:expr, $sel:expr, $($arg:expr),+) => {{
        let sel = sel_registerName(concat!($sel, "\0").as_ptr());
        objc_msgSend($obj, sel, $($arg),+)
    }};
}

// ── Delegate class registration ─────────────────────────────────────

/// The delegate's `_frameSender` ivar stores a raw pointer to
/// `Box<mpsc::Sender<VideoFrame>>`. It is set when the delegate is allocated
/// and freed when the camera source is dropped.
static DELEGATE_CLASS_INIT: std::sync::Once = std::sync::Once::new();
static mut DELEGATE_CLASS: Class = ptr::null_mut();

fn ensure_delegate_class() {
    DELEGATE_CLASS_INIT.call_once(|| unsafe {
        let superclass = objc_getClass(b"NSObject\0".as_ptr());
        let cls = objc_allocateClassPair(superclass, b"AlexandriaFrameDelegate\0".as_ptr(), 0);
        assert!(!cls.is_null(), "Failed to allocate ObjC delegate class");

        // Add ivar: void *_frameSender
        class_addIvar(
            cls,
            b"_frameSender\0".as_ptr(),
            std::mem::size_of::<*mut c_void>(),
            std::mem::align_of::<*mut c_void>() as u8,
            b"^v\0".as_ptr(),
        );

        // Add method: captureOutput:didOutputSampleBuffer:fromConnection:
        class_addMethod(
            cls,
            sel_registerName(b"captureOutput:didOutputSampleBuffer:fromConnection:\0".as_ptr()),
            delegate_callback as IMP,
            b"v@:@@@\0".as_ptr(), // return void, self, _cmd, 3 id args
        );

        objc_registerClassPair(cls);
        DELEGATE_CLASS = cls;
    });
}

/// ObjC method implementation for the delegate callback.
///
/// # Safety
/// Called by AVFoundation on its internal dispatch queue.
unsafe extern "C" fn delegate_callback(
    this: Id,
    _cmd: Sel,
    _output: Id,       // AVCaptureOutput
    sample_buffer: Id, // CMSampleBuffer
    _connection: Id,   // AVCaptureConnection
) {
    if sample_buffer.is_null() {
        return;
    }

    // Get the sender from the ivar
    let mut sender_ptr: *mut c_void = ptr::null_mut();
    object_getInstanceVariable(this, b"_frameSender\0".as_ptr(), &mut sender_ptr);
    if sender_ptr.is_null() {
        return;
    }
    let sender = &*(sender_ptr as *const mpsc::Sender<VideoFrame>);

    // Get pixel buffer from sample buffer
    let pixel_buffer = CMSampleBufferGetImageBuffer(sample_buffer as CMSampleBufferRef);
    if pixel_buffer.is_null() {
        return;
    }

    // Lock and read BGRA data
    let lock_status = CVPixelBufferLockBaseAddress(pixel_buffer, 1); // read-only
    if lock_status != 0 {
        return;
    }

    let base = CVPixelBufferGetBaseAddress(pixel_buffer);
    let bytes_per_row = CVPixelBufferGetBytesPerRow(pixel_buffer);
    let width = CVPixelBufferGetWidth(pixel_buffer) as u32;
    let height = CVPixelBufferGetHeight(pixel_buffer) as u32;

    if !base.is_null() && width > 0 && height > 0 {
        // Copy BGRA data (handle stride != width*4)
        let expected_row = width as usize * 4;
        let data = if bytes_per_row == expected_row {
            let total = expected_row * height as usize;
            std::slice::from_raw_parts(base as *const u8, total).to_vec()
        } else {
            // Row-by-row copy to strip padding
            let mut buf = Vec::with_capacity(expected_row * height as usize);
            for y in 0..height as usize {
                let row_ptr = (base as *const u8).add(y * bytes_per_row);
                let row = std::slice::from_raw_parts(row_ptr, expected_row);
                buf.extend_from_slice(row);
            }
            buf
        };

        let frame = VideoFrame {
            format: VideoFormat {
                pixel_format: PixelFormat::Bgra,
                dimensions: [width, height],
            },
            raw: Bytes::from(data),
        };

        // Non-blocking send — drop frame if receiver is behind
        let _ = sender.try_send(frame);
    }

    CVPixelBufferUnlockBaseAddress(pixel_buffer, 1);
}

// ── IosCameraSource ─────────────────────────────────────────────────

/// iOS camera video source using AVCaptureSession.
///
/// Captures BGRA frames from the device camera and delivers them via
/// the `VideoSource` trait.
pub struct IosCameraSource {
    session: Id,  // AVCaptureSession (retained)
    delegate: Id, // AlexandriaFrameDelegate (retained)
    rx: mpsc::Receiver<VideoFrame>,
    /// Leaked Box<Sender> — freed on drop.
    sender_ptr: *mut mpsc::Sender<VideoFrame>,
    width: u32,
    height: u32,
    running: bool,
}

// SAFETY: AVCaptureSession operations are dispatched on serial queues;
// our owned references are only accessed from one thread at a time.
unsafe impl Send for IosCameraSource {}

impl IosCameraSource {
    /// Create a new camera source using the back camera (front as fallback).
    ///
    /// Resolution defaults to 640x480 (AVCaptureSessionPresetMedium).
    pub fn new() -> Result<Self> {
        Self::with_position(1) // 1 = AVCaptureDevicePositionBack
    }

    /// Create using front camera.
    pub fn front() -> Result<Self> {
        Self::with_position(2) // 2 = AVCaptureDevicePositionFront
    }

    fn with_position(position: isize) -> Result<Self> {
        ensure_delegate_class();

        unsafe {
            // Create AVCaptureSession
            let session_class = objc_getClass(b"AVCaptureSession\0".as_ptr());
            if session_class.is_null() {
                bail!("AVCaptureSession class not found");
            }
            let session: Id = msg_send!(msg_send!(session_class, "alloc"), "init");
            if session.is_null() {
                bail!("Failed to create AVCaptureSession");
            }

            // Set preset to Medium (640x480)
            let preset_sel = sel_registerName(b"setSessionPreset:\0".as_ptr());
            // AVCaptureSessionPresetMedium is an NSString constant
            let preset_str = get_nsstring("AVCaptureSessionPresetMedium");
            if !preset_str.is_null() {
                objc_msgSend(session, preset_sel, preset_str);
            }

            // Get camera device
            let device = find_camera_device(position);
            if device.is_null() {
                // Fallback: try the other position
                let fallback_pos = if position == 1 { 2 } else { 1 };
                let device = find_camera_device(fallback_pos);
                if device.is_null() {
                    release(session);
                    bail!("No camera device found");
                }
                Self::setup_session(session, device)
            } else {
                Self::setup_session(session, device)
            }
        }
    }

    unsafe fn setup_session(session: Id, device: Id) -> Result<Self> {
        // Create AVCaptureDeviceInput
        let input_class = objc_getClass(b"AVCaptureDeviceInput\0".as_ptr());
        let mut error: Id = NIL;
        let input: Id = {
            let sel = sel_registerName(b"deviceInputWithDevice:error:\0".as_ptr());
            objc_msgSend(input_class, sel, device, &mut error as *mut Id)
        };
        if input.is_null() || !error.is_null() {
            release(session);
            bail!("Failed to create AVCaptureDeviceInput");
        }

        // Add input
        let can_add_input: BOOL = {
            let sel = sel_registerName(b"canAddInput:\0".as_ptr());
            objc_msgSend(session, sel, input) as BOOL
        };
        if can_add_input != YES {
            release(session);
            bail!("Cannot add camera input to session");
        }
        {
            let sel = sel_registerName(b"addInput:\0".as_ptr());
            objc_msgSend(session, sel, input);
        }

        // Create AVCaptureVideoDataOutput
        let output_class = objc_getClass(b"AVCaptureVideoDataOutput\0".as_ptr());
        let output: Id = msg_send!(msg_send!(output_class, "alloc"), "init");
        if output.is_null() {
            release(session);
            bail!("Failed to create AVCaptureVideoDataOutput");
        }

        // Set pixel format to BGRA
        let settings = create_pixel_format_settings(K_CV_PIXEL_FORMAT_TYPE_32_BGRA);
        if !settings.is_null() {
            let sel = sel_registerName(b"setVideoSettings:\0".as_ptr());
            objc_msgSend(output, sel, settings);
        }

        // Discard late frames
        {
            let sel = sel_registerName(b"setAlwaysDiscardsLateVideoFrames:\0".as_ptr());
            objc_msgSend(output, sel, YES);
        }

        // Create delegate
        let delegate: Id = msg_send!(msg_send!(DELEGATE_CLASS, "alloc"), "init");
        if delegate.is_null() {
            release(output);
            release(session);
            bail!("Failed to create frame delegate");
        }

        // Create channel for frames
        let (tx, rx) = mpsc::sync_channel::<VideoFrame>(2);
        let sender_box = Box::new(tx);
        let sender_ptr = Box::into_raw(sender_box);

        // Store sender in delegate ivar
        object_setInstanceVariable(
            delegate,
            b"_frameSender\0".as_ptr(),
            sender_ptr as *mut c_void,
        );

        // Set delegate with a serial dispatch queue
        let queue = create_dispatch_queue(b"org.alexandria.camera\0".as_ptr());
        {
            let sel = sel_registerName(b"setSampleBufferDelegate:queue:\0".as_ptr());
            objc_msgSend(output, sel, delegate, queue);
        }

        // Add output
        let can_add_output: BOOL = {
            let sel = sel_registerName(b"canAddOutput:\0".as_ptr());
            objc_msgSend(session, sel, output) as BOOL
        };
        if can_add_output != YES {
            release(delegate);
            release(output);
            release(session);
            let _ = Box::from_raw(sender_ptr); // reclaim
            bail!("Cannot add video output to session");
        }
        {
            let sel = sel_registerName(b"addOutput:\0".as_ptr());
            objc_msgSend(session, sel, output);
        }

        // We can release output — session retains it
        release(output);

        Ok(IosCameraSource {
            session,
            delegate,
            rx,
            sender_ptr,
            width: 640,
            height: 480,
            running: false,
        })
    }
}

impl Drop for IosCameraSource {
    fn drop(&mut self) {
        unsafe {
            if self.running {
                msg_send!(self.session, "stopRunning");
            }

            // Clear delegate ivar to prevent dangling pointer
            object_setInstanceVariable(self.delegate, b"_frameSender\0".as_ptr(), ptr::null_mut());

            // Reclaim the leaked sender
            if !self.sender_ptr.is_null() {
                let _ = Box::from_raw(self.sender_ptr);
                self.sender_ptr = ptr::null_mut();
            }

            release(self.delegate);
            release(self.session);
        }
    }
}

impl VideoSource for IosCameraSource {
    fn name(&self) -> &str {
        "ios-camera"
    }

    fn format(&self) -> VideoFormat {
        VideoFormat {
            pixel_format: PixelFormat::Bgra,
            dimensions: [self.width, self.height],
        }
    }

    fn start(&mut self) -> Result<()> {
        if self.running {
            return Ok(());
        }
        unsafe {
            msg_send!(self.session, "startRunning");
        }
        self.running = true;
        tracing::info!("iOS camera started ({}x{})", self.width, self.height);
        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        if !self.running {
            return Ok(());
        }
        unsafe {
            msg_send!(self.session, "stopRunning");
        }
        self.running = false;
        // Drain any pending frames
        while self.rx.try_recv().is_ok() {}
        tracing::info!("iOS camera stopped");
        Ok(())
    }

    fn pop_frame(&mut self) -> Result<Option<VideoFrame>> {
        // Drain to latest frame (non-blocking)
        let mut latest = None;
        while let Ok(frame) = self.rx.try_recv() {
            latest = Some(frame);
        }
        Ok(latest)
    }
}

// ── ObjC helper functions ───────────────────────────────────────────

unsafe fn release(obj: Id) {
    if !obj.is_null() {
        msg_send!(obj, "release");
    }
}

/// Find a camera device by position (1=back, 2=front).
unsafe fn find_camera_device(position: isize) -> Id {
    let device_class = objc_getClass(b"AVCaptureDevice\0".as_ptr());
    if device_class.is_null() {
        return NIL;
    }

    // Try AVCaptureDevice.defaultDeviceWithDeviceType:mediaType:position:
    let device_type = get_nsstring("AVCaptureDeviceTypeBuiltInWideAngleCamera");
    let media_type = get_nsstring("AVMediaTypeVideo");

    if !device_type.is_null() && !media_type.is_null() {
        let sel = sel_registerName(b"defaultDeviceWithDeviceType:mediaType:position:\0".as_ptr());
        let device: Id = objc_msgSend(device_class, sel, device_type, media_type, position);
        if !device.is_null() {
            return device;
        }
    }

    // Fallback: AVCaptureDevice.defaultDeviceWithMediaType:
    if position == 1 || position == 2 {
        let sel = sel_registerName(b"defaultDeviceWithMediaType:\0".as_ptr());
        let avmedia_video = get_nsstring("AVMediaTypeVideo");
        if !avmedia_video.is_null() {
            let device: Id = objc_msgSend(device_class, sel, avmedia_video);
            if !device.is_null() {
                return device;
            }
        }
    }

    NIL
}

/// Get an NSString constant by looking it up from the AVFoundation framework symbol.
/// Falls back to creating an NSString from a UTF-8 literal.
unsafe fn get_nsstring(name: &str) -> Id {
    // Create NSString from literal
    let nsstring_class = objc_getClass(b"NSString\0".as_ptr());
    if nsstring_class.is_null() {
        return NIL;
    }
    let c_str = std::ffi::CString::new(name).unwrap_or_default();
    let sel = sel_registerName(b"stringWithUTF8String:\0".as_ptr());
    objc_msgSend(nsstring_class, sel, c_str.as_ptr())
}

/// Create an NSDictionary with kCVPixelBufferPixelFormatTypeKey → pixel format.
unsafe fn create_pixel_format_settings(pixel_format: u32) -> Id {
    let nsnum_class = objc_getClass(b"NSNumber\0".as_ptr());
    let nsdict_class = objc_getClass(b"NSDictionary\0".as_ptr());
    if nsnum_class.is_null() || nsdict_class.is_null() {
        return NIL;
    }

    let num: Id = {
        let sel = sel_registerName(b"numberWithUnsignedInt:\0".as_ptr());
        objc_msgSend(nsnum_class, sel, pixel_format)
    };

    // Key: kCVPixelBufferPixelFormatTypeKey (which is the NSString "PixelFormatType")
    let key = get_nsstring("PixelFormatType");
    if key.is_null() || num.is_null() {
        return NIL;
    }

    let sel = sel_registerName(b"dictionaryWithObject:forKey:\0".as_ptr());
    objc_msgSend(nsdict_class, sel, num, key)
}

// GCD dispatch queue creation
unsafe extern "C" {
    fn dispatch_queue_create(label: *const u8, attr: Id) -> Id;
}

/// Create a serial dispatch queue for camera callbacks.
unsafe fn create_dispatch_queue(label: *const u8) -> Id {
    dispatch_queue_create(label, NIL) // NULL attr = serial
}
