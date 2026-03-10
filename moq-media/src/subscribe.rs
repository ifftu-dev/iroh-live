use std::{collections::BTreeMap, sync::Arc, time::Duration};

use hang::{
    Timestamp, TrackConsumer,
    catalog::{AudioConfig, Catalog, CatalogConsumer, VideoConfig},
};
use moq_lite::{BroadcastConsumer, Track};
use n0_error::{Result, StackResultExt, StdResultExt};
use n0_future::task::AbortOnDropHandle;
use n0_watcher::{Watchable, Watcher};
use tokio::{
    sync::mpsc::{self, error::TryRecvError},
    time::Instant,
};
use tokio_util::sync::{CancellationToken, DropGuard};
use tracing::{Span, debug, error, info, info_span, trace, warn};

use crate::{
    av::{AudioDecoder, AudioSink, AudioSinkHandle, Quality},
    util::spawn_thread,
};
#[cfg(any(feature = "video", feature = "video-ios"))]
use crate::av::{DecodeConfig, DecodedFrame, Decoders, PlaybackConfig, VideoDecoder};
#[cfg(any(feature = "video", feature = "video-ios"))]
use crate::av::VideoSource;
#[cfg(feature = "video")]
use crate::ffmpeg::util::Rescaler;

const DEFAULT_MAX_LATENCY: Duration = Duration::from_millis(150);

#[derive(derive_more::Debug, Clone)]
pub struct SubscribeBroadcast {
    broadcast_name: String,
    #[debug("BroadcastConsumer")]
    broadcast: BroadcastConsumer,
    // catalog_watcher: n0_watcher::Direct<CatalogWrapper>,
    catalog_watchable: Watchable<CatalogWrapper>,
    shutdown: CancellationToken,
    _catalog_task: Arc<AbortOnDropHandle<()>>,
}

#[derive(Debug, derive_more::PartialEq, derive_more::Eq, Default, Clone, derive_more::Deref)]
pub struct CatalogWrapper {
    #[eq(skip)]
    #[deref]
    inner: Arc<Catalog>,
    seq: usize,
}

impl CatalogWrapper {
    fn new(inner: Catalog, seq: usize) -> Self {
        Self {
            inner: Arc::new(inner),
            seq,
        }
    }

    #[cfg(any(feature = "video", feature = "video-ios"))]
    pub fn video_renditions(&self) -> impl Iterator<Item = &str> {
        let mut renditions: Vec<_> = self
            .inner
            .video
            .as_ref()
            .iter()
            .map(|v| v.renditions.iter())
            .flatten()
            .map(|(name, config)| (name.as_str(), config.coded_width))
            .collect();
        renditions.sort_by(|a, b| a.1.cmp(&b.1));
        renditions.into_iter().map(|(name, _w)| name)
    }

    pub fn audio_renditions(&self) -> impl Iterator<Item = &str> + '_ {
        self.inner
            .audio
            .as_ref()
            .into_iter()
            .map(|v| v.renditions.iter())
            .flatten()
            .map(|(name, _config)| name.as_str())
    }

    #[cfg(any(feature = "video", feature = "video-ios"))]
    pub fn select_video_rendition(&self, quality: Quality) -> Result<String> {
        let video = self.inner.video.as_ref().context("no video published")?;
        let track_name =
            select_video_rendition(&video.renditions, quality).context("no video renditions")?;
        Ok(track_name)
    }

    pub fn select_audio_rendition(&self, quality: Quality) -> Result<String> {
        let audio = self.inner.audio.as_ref().context("no video published")?;
        let track_name =
            select_audio_rendition(&audio.renditions, quality).context("no video renditions")?;
        Ok(track_name)
    }
}

impl CatalogWrapper {
    pub fn into_inner(self) -> Arc<Catalog> {
        self.inner
    }
}

impl SubscribeBroadcast {
    pub async fn new(broadcast_name: String, broadcast: BroadcastConsumer) -> Result<Self> {
        let shutdown = CancellationToken::new();

        let (catalog_watchable, catalog_task) = {
            let track = broadcast.subscribe_track(&Catalog::default_track());
            let mut consumer = CatalogConsumer::new(track);
            let initial_catalog = consumer
                .next()
                .await
                .std_context("Broadcast closed before receiving catalog")?
                .context("Catalog track closed before receiving catalog")?;
            let watchable = Watchable::new(CatalogWrapper::new(initial_catalog, 0));

            let task = tokio::spawn({
                let shutdown = shutdown.clone();
                let watchable = watchable.clone();
                async move {
                    for seq in 1.. {
                        match consumer.next().await {
                            Ok(Some(catalog)) => {
                                watchable.set(CatalogWrapper::new(catalog, seq)).ok();
                            }
                            Ok(None) => {
                                debug!("subscribed broadcast catalog track ended");
                                break;
                            }
                            Err(err) => {
                                debug!("subscribed broadcast closed: {err:#}");
                                break;
                            }
                        }
                    }
                    shutdown.cancel();
                }
            });
            (watchable, task)
        };
        Ok(Self {
            broadcast_name,
            broadcast,
            catalog_watchable,
            _catalog_task: Arc::new(AbortOnDropHandle::new(catalog_task)),
            shutdown: CancellationToken::new(),
        })
    }

    pub fn broadcast_name(&self) -> &str {
        &self.broadcast_name
    }

    pub fn catalog_watcher(&mut self) -> n0_watcher::Direct<CatalogWrapper> {
        self.catalog_watchable.watch()
    }

    pub fn catalog(&self) -> CatalogWrapper {
        self.catalog_watchable.get()
    }

    #[cfg(any(feature = "video", feature = "video-ios"))]
    pub fn watch_and_listen<D: Decoders>(
        self,
        audio_out: impl AudioSink,
        playback_config: PlaybackConfig,
    ) -> Result<AvRemoteTrack> {
        AvRemoteTrack::new::<D>(self, audio_out, playback_config)
    }

    #[cfg(any(feature = "video", feature = "video-ios"))]
    pub fn watch<D: VideoDecoder>(&self) -> Result<WatchTrack> {
        self.watch_with::<D>(&Default::default(), Quality::Highest)
    }

    #[cfg(any(feature = "video", feature = "video-ios"))]
    pub fn watch_with<D: VideoDecoder>(
        &self,
        playback_config: &DecodeConfig,
        quality: Quality,
    ) -> Result<WatchTrack> {
        let track_name = self.catalog().select_video_rendition(quality)?;
        self.watch_rendition::<D>(playback_config, &track_name)
    }

    #[cfg(any(feature = "video", feature = "video-ios"))]
    pub fn watch_rendition<D: VideoDecoder>(
        &self,
        playback_config: &DecodeConfig,
        track_name: &str,
    ) -> Result<WatchTrack> {
        let catalog = self.catalog();
        let video = catalog.video.as_ref().context("no video published")?;
        let config = video
            .renditions
            .get(track_name)
            .context("rendition not found")?;
        let consumer = TrackConsumer::new(
            self.broadcast.subscribe_track(&Track {
                name: track_name.to_string(),
                priority: video.priority,
            }),
            DEFAULT_MAX_LATENCY,
        );
        let span = info_span!("videodec", %track_name);
        WatchTrack::from_consumer::<D>(
            track_name.to_string(),
            consumer,
            &config,
            playback_config,
            self.shutdown.child_token(),
            span,
        )
    }
    pub fn listen<D: AudioDecoder>(&self, output: impl AudioSink) -> Result<AudioTrack> {
        self.listen_with::<D>(Quality::Highest, output)
    }

    pub fn listen_with<D: AudioDecoder>(
        &self,
        quality: Quality,
        output: impl AudioSink,
    ) -> Result<AudioTrack> {
        let track_name = self.catalog().select_audio_rendition(quality)?;
        self.listen_rendition::<D>(&track_name, output)
    }

    pub fn listen_rendition<D: AudioDecoder>(
        &self,
        name: &str,
        output: impl AudioSink,
    ) -> Result<AudioTrack> {
        let catalog = self.catalog();
        let audio = catalog.audio.as_ref().context("no audio published")?;
        let config = audio.renditions.get(name).context("rendition not found")?;
        let consumer = TrackConsumer::new(
            self.broadcast.subscribe_track(&Track {
                name: name.to_string(),
                priority: audio.priority,
            }),
            DEFAULT_MAX_LATENCY,
        );
        let span = info_span!("audiodec", %name);
        AudioTrack::spawn::<D>(
            name.to_string(),
            consumer,
            config.clone(),
            output,
            self.shutdown.child_token(),
            span,
        )
    }

    pub fn closed(&self) -> impl Future<Output = ()> + 'static {
        self.broadcast.closed()
    }

    pub fn shutdown(&self) {
        self.shutdown.cancel();
    }
}

fn select_rendition<T, P: ToString>(
    renditions: &BTreeMap<String, T>,
    order: &[P],
) -> Option<String> {
    order
        .iter()
        .map(ToString::to_string)
        .find(|k| renditions.contains_key(k.as_str()))
        .or_else(|| renditions.keys().next().cloned())
}

#[cfg(any(feature = "video", feature = "video-ios"))]
fn select_video_rendition<'a, T>(
    renditions: &'a BTreeMap<String, T>,
    q: Quality,
) -> Option<String> {
    use crate::av::VideoPreset::*;
    let order = match q {
        Quality::Highest => [P1080, P720, P360, P180],
        Quality::High => [P720, P360, P180, P1080],
        Quality::Mid => [P360, P180, P720, P1080],
        Quality::Low => [P180, P360, P720, P1080],
    };

    select_rendition(renditions, &order)
}

fn select_audio_rendition<'a, T>(
    renditions: &'a BTreeMap<String, T>,
    q: Quality,
) -> Option<String> {
    use crate::av::AudioPreset::*;
    let order = match q {
        Quality::Highest | Quality::High => [Hq, Lq],
        Quality::Mid | Quality::Low => [Lq, Hq],
    };
    select_rendition(renditions, &order)
}

pub struct AudioTrack {
    name: String,
    handle: Box<dyn AudioSinkHandle>,
    shutdown_token: CancellationToken,
    _task_handle: AbortOnDropHandle<()>,
    _thread_handle: std::thread::JoinHandle<()>,
}

impl AudioTrack {
    pub(crate) fn spawn<D: AudioDecoder>(
        name: String,
        consumer: TrackConsumer,
        config: AudioConfig,
        output: impl AudioSink,
        shutdown: CancellationToken,
        span: Span,
    ) -> Result<Self> {
        let _guard = span.enter();
        let (packet_tx, packet_rx) = mpsc::channel(32);
        let output_format = output.format()?;
        info!(?config, "audio thread start");
        let decoder = D::new(&config, output_format)?;
        let handle = output.handle();
        let thread_name = format!("adec-{}", name);
        let thread = spawn_thread(thread_name, {
            let shutdown = shutdown.clone();
            let span = span.clone();
            move || {
                let _guard = span.enter();
                if let Err(err) = Self::run_loop(decoder, packet_rx, output, &shutdown) {
                    error!("audio decoder failed: {err:#}");
                }
                info!("audio decoder thread stop");
            }
        });
        let task = tokio::spawn(forward_frames(consumer, packet_tx));
        Ok(Self {
            name,
            handle,
            shutdown_token: shutdown,
            _task_handle: AbortOnDropHandle::new(task),
            _thread_handle: thread,
        })
    }

    pub fn stopped(&self) -> impl Future<Output = ()> + 'static {
        let shutdown_token = self.shutdown_token.clone();
        async move { shutdown_token.cancelled().await }
    }

    pub fn rendition(&self) -> &str {
        &self.name
    }

    pub fn handle(&self) -> &dyn AudioSinkHandle {
        self.handle.as_ref()
    }

    pub(crate) fn run_loop(
        mut decoder: impl AudioDecoder,
        mut packet_rx: mpsc::Receiver<hang::Frame>,
        mut sink: impl AudioSink,
        shutdown: &CancellationToken,
    ) -> Result<()> {
        const INTERVAL: Duration = Duration::from_millis(10);
        let mut remote_start = None;
        let loop_start = Instant::now();
        let mut audio_pkt_count: u64 = 0;
        info!("audiodec: run_loop starting, waiting for packets...");

        'main: for i in 0.. {
            let tick = Instant::now();

            if shutdown.is_cancelled() {
                info!("audiodec: cancelled after {audio_pkt_count} packets");
                break;
            }

            loop {
                match packet_rx.try_recv() {
                    Ok(packet) => {
                        audio_pkt_count += 1;
                        if audio_pkt_count <= 3 || audio_pkt_count % 500 == 0 {
                            info!(
                                "audiodec: packet #{audio_pkt_count}, bytes={}, paused={}",
                                packet.payload.num_bytes(),
                                sink.is_paused()
                            );
                        }
                        let remote_start = *remote_start.get_or_insert_with(|| packet.timestamp);

                        if tracing::enabled!(tracing::Level::TRACE) {
                            let loop_elapsed = tick.duration_since(loop_start);
                            let remote_elapsed: Duration = packet
                                .timestamp
                                .checked_sub(remote_start)
                                .unwrap_or(Timestamp::ZERO)
                                .into();
                            let diff_ms =
                                (loop_elapsed.as_secs_f32() - remote_elapsed.as_secs_f32()) * 1000.;
                            trace!(len = packet.payload.num_bytes(), ts=?packet.timestamp, ?loop_elapsed, ?remote_elapsed, ?diff_ms, "recv packet");
                        }

                        // TODO: Skip outdated packets?

                        if !sink.is_paused() {
                            decoder.push_packet(packet)?;
                            if let Some(samples) = decoder.pop_samples()? {
                                sink.push_samples(samples)?;
                            }
                        }
                    }
                    Err(TryRecvError::Disconnected) => {
                        info!("audiodec: disconnected after {audio_pkt_count} packets");
                        break 'main;
                    }
                    Err(TryRecvError::Empty) => {
                        trace!("no packet to recv");
                        break;
                    }
                }
            }

            let target_time = i * INTERVAL;
            let real_time = Instant::now().duration_since(loop_start);
            let sleep = target_time.saturating_sub(real_time);
            if !sleep.is_zero() {
                std::thread::sleep(sleep);
            }
        }
        shutdown.cancel();
        Ok(())
    }
}

impl Drop for AudioTrack {
    fn drop(&mut self) {
        self.shutdown_token.cancel();
    }
}

#[cfg(any(feature = "video", feature = "video-ios"))]
pub struct WatchTrack {
    video_frames: WatchTrackFrames,
    handle: WatchTrackHandle,
}

#[cfg(any(feature = "video", feature = "video-ios"))]
pub struct WatchTrackHandle {
    rendition: String,
    viewport: Watchable<(u32, u32)>,
    _guard: WatchTrackGuard,
}

#[cfg(any(feature = "video", feature = "video-ios"))]
impl WatchTrackHandle {
    pub fn set_viewport(&self, w: u32, h: u32) {
        self.viewport.set((w, h)).ok();
    }

    pub fn rendition(&self) -> &str {
        &self.rendition
    }
}

#[cfg(any(feature = "video", feature = "video-ios"))]
pub struct WatchTrackFrames {
    rx: mpsc::Receiver<DecodedFrame>,
}

#[cfg(any(feature = "video", feature = "video-ios"))]
impl WatchTrackFrames {
    pub fn current_frame(&mut self) -> Option<DecodedFrame> {
        let mut out = None;
        while let Ok(item) = self.rx.try_recv() {
            out = Some(item);
        }
        out
    }

    pub async fn next_frame(&mut self) -> Option<DecodedFrame> {
        if let Some(frame) = self.current_frame() {
            Some(frame)
        } else {
            self.rx.recv().await
        }
    }
}

#[cfg(any(feature = "video", feature = "video-ios"))]
struct WatchTrackGuard {
    _shutdown_token_guard: DropGuard,
    _task_handle: Option<AbortOnDropHandle<()>>,
    _thread_handle: Option<std::thread::JoinHandle<()>>,
}

#[cfg(any(feature = "video", feature = "video-ios"))]
impl WatchTrack {
    pub fn empty(rendition: impl ToString) -> Self {
        let (tx, rx) = mpsc::channel(1);
        let task = tokio::task::spawn(async move {
            std::future::pending::<()>().await;
            let _ = tx;
        });
        let guard = WatchTrackGuard {
            _shutdown_token_guard: CancellationToken::new().drop_guard(),
            _task_handle: Some(AbortOnDropHandle::new(task)),
            _thread_handle: None,
        };
        Self {
            video_frames: WatchTrackFrames { rx },
            handle: WatchTrackHandle {
                rendition: rendition.to_string(),
                viewport: Default::default(),
                _guard: guard,
            },
        }
    }

    /// Local video preview — requires ffmpeg Rescaler (desktop only).
    #[cfg(feature = "video")]
    pub fn from_video_source(
        rendition: String,
        shutdown: CancellationToken,
        mut source: impl VideoSource,
        decode_config: DecodeConfig,
    ) -> Self {
        let viewport = Watchable::new((1u32, 1u32));
        let (frame_tx, frame_rx) = tokio::sync::mpsc::channel::<DecodedFrame>(2);
        let thread_name = format!("vpr-{:>4}-{:>4}", source.name(), rendition);
        let thread = spawn_thread(thread_name, {
            let mut viewport = viewport.watch();
            let shutdown = shutdown.clone();
            move || {
                // TODO: Make configurable.
                let fps = 30;
                let mut rescaler = Rescaler::new(decode_config.pixel_format.to_ffmpeg(), None)
                    .expect("failed to create rescaler");
                let frame_duration = Duration::from_secs_f32(1. / fps as f32);
                if let Err(err) = source.start() {
                    warn!("Video source failed to start: {err:?}");
                    return;
                }
                let start = Instant::now();
                for i in 1.. {
                    // let t = Instant::now();
                    if shutdown.is_cancelled() {
                        break;
                    }
                    if viewport.update() {
                        let (w, h) = viewport.peek();
                        rescaler.set_target_dimensions(*w, *h);
                    }
                    match source.pop_frame() {
                        Ok(Some(frame)) => {
                            // trace!(t=?t.elapsed(), "pop");
                            let frame = frame.to_ffmpeg();
                            let frame = rescaler.process(&frame).expect("rescaler failed");
                            let frame =
                                DecodedFrame::from_ffmpeg(frame, frame_duration, start.elapsed());
                            // trace!(t=?t.elapsed(), "convert");
                            let _ = frame_tx.blocking_send(frame);
                            // trace!(t=?t.elapsed(), "send");
                        }
                        Ok(None) => {}
                        Err(_) => break,
                    }
                    let expected_time = i * frame_duration;
                    let actual_time = start.elapsed();
                    if expected_time > actual_time {
                        std::thread::sleep(expected_time - actual_time);
                        // trace!(t=?t.elapsed(), slept=?(actual_time - expected_time), ?expected_time, ?actual_time, "done");
                    }
                }
                if let Err(err) = source.stop() {
                    warn!("Video source failed to stop: {err:?}");
                    return;
                }
            }
        });
        let guard = WatchTrackGuard {
            _shutdown_token_guard: shutdown.drop_guard(),
            _task_handle: None,
            _thread_handle: Some(thread),
        };
        WatchTrack {
            video_frames: WatchTrackFrames { rx: frame_rx },
            handle: WatchTrackHandle {
                rendition,
                viewport,
                _guard: guard,
            },
        }
    }

    /// Local video preview for iOS — no ffmpeg dependency.
    ///
    /// Converts raw camera frames (BGRA) to RGBA directly, without
    /// ffmpeg's Rescaler. Suitable for self-preview where the camera
    /// resolution is already appropriate (640×480 from AVCaptureSession).
    #[cfg(feature = "video-ios")]
    pub fn from_video_source_raw(
        rendition: String,
        shutdown: CancellationToken,
        mut source: impl VideoSource,
    ) -> Self {
        use crate::av::PixelFormat;

        let viewport = Watchable::new((1u32, 1u32));
        let (frame_tx, frame_rx) = tokio::sync::mpsc::channel::<DecodedFrame>(2);
        let thread_name = format!("vpr-{:>4}-{:>4}", source.name(), rendition);
        let thread = spawn_thread(thread_name, {
            let shutdown = shutdown.clone();
            move || {
                let fps = 24;
                let frame_duration = Duration::from_secs_f32(1. / fps as f32);
                if let Err(err) = source.start() {
                    warn!("Video source failed to start: {err:?}");
                    return;
                }
                let start = std::time::Instant::now();
                for i in 1u64.. {
                    if shutdown.is_cancelled() {
                        break;
                    }
                    match source.pop_frame() {
                        Ok(Some(frame)) => {
                            let [w, h] = frame.format.dimensions;
                            let raw = &frame.raw;

                            // Convert BGRA → RGBA if needed (swap B and R channels)
                            let rgba_data = match frame.format.pixel_format {
                                PixelFormat::Bgra => {
                                    let mut rgba = Vec::with_capacity(raw.len());
                                    for chunk in raw.chunks_exact(4) {
                                        rgba.push(chunk[2]); // R (was B)
                                        rgba.push(chunk[1]); // G
                                        rgba.push(chunk[0]); // B (was R)
                                        rgba.push(chunk[3]); // A
                                    }
                                    rgba
                                }
                                PixelFormat::Rgba => raw.to_vec(),
                            };

                            if let Some(img) = image::RgbaImage::from_raw(w, h, rgba_data) {
                                let decoded = DecodedFrame {
                                    frame: image::Frame::new(img),
                                    timestamp: start.elapsed(),
                                };
                                let _ = frame_tx.blocking_send(decoded);
                            }
                        }
                        Ok(None) => {}
                        Err(_) => break,
                    }
                    let expected_time = frame_duration * i as u32;
                    let actual_time = start.elapsed();
                    if expected_time > actual_time {
                        std::thread::sleep(expected_time - actual_time);
                    }
                }
                if let Err(err) = source.stop() {
                    warn!("Video source failed to stop: {err:?}");
                }
            }
        });
        let guard = WatchTrackGuard {
            _shutdown_token_guard: shutdown.drop_guard(),
            _task_handle: None,
            _thread_handle: Some(thread),
        };
        WatchTrack {
            video_frames: WatchTrackFrames { rx: frame_rx },
            handle: WatchTrackHandle {
                rendition,
                viewport,
                _guard: guard,
            },
        }
    }

    pub(crate) fn from_consumer<D: VideoDecoder>(
        rendition: String,
        consumer: TrackConsumer,
        config: &VideoConfig,
        playback_config: &DecodeConfig,
        shutdown: CancellationToken,
        span: Span,
    ) -> Result<Self> {
        let (packet_tx, packet_rx) = mpsc::channel(32);
        let (frame_tx, frame_rx) = mpsc::channel(32);
        let viewport = Watchable::new((1u32, 1u32));
        let viewport_watcher = viewport.watch();

        let _guard = span.enter();
        debug!(?config, "video decoder start");
        let decoder = D::new(config, playback_config)?;
        let thread_name = format!("vdec-{}", rendition);
        let thread = spawn_thread(thread_name, {
            let shutdown = shutdown.clone();
            let span = span.clone();
            move || {
                let _guard = span.enter();
                if let Err(err) =
                    Self::run_loop(&shutdown, packet_rx, frame_tx, viewport_watcher, decoder)
                {
                    error!("video decoder failed: {err:#}");
                }
                shutdown.cancel();
            }
        });
        let task = tokio::task::spawn(forward_frames(consumer, packet_tx));
        let guard = WatchTrackGuard {
            _shutdown_token_guard: shutdown.drop_guard(),
            _task_handle: Some(AbortOnDropHandle::new(task)),
            _thread_handle: Some(thread),
        };
        Ok(WatchTrack {
            video_frames: WatchTrackFrames { rx: frame_rx },
            handle: WatchTrackHandle {
                rendition,
                viewport,
                _guard: guard,
            },
        })
    }

    pub fn split(self) -> (WatchTrackFrames, WatchTrackHandle) {
        (self.video_frames, self.handle)
    }

    pub fn set_viewport(&self, w: u32, h: u32) {
        self.handle.set_viewport(w, h);
    }

    pub fn rendition(&self) -> &str {
        self.handle.rendition()
    }

    pub fn current_frame(&mut self) -> Option<DecodedFrame> {
        self.video_frames.current_frame()
    }

    pub(crate) fn run_loop(
        shutdown: &CancellationToken,
        mut input_rx: mpsc::Receiver<hang::Frame>,
        output_tx: mpsc::Sender<DecodedFrame>,
        mut viewport_watcher: n0_watcher::Direct<(u32, u32)>,
        mut decoder: impl VideoDecoder,
    ) -> Result<(), anyhow::Error> {
        let mut pkt_count: u64 = 0;
        let mut decoded_count: u64 = 0;
        info!("videodec: run_loop starting, waiting for first packet...");
        loop {
            if shutdown.is_cancelled() {
                info!("videodec: run_loop cancelled after {pkt_count} packets, {decoded_count} decoded frames");
                break;
            }
            let Some(packet) = input_rx.blocking_recv() else {
                info!("videodec: run_loop input channel closed after {pkt_count} packets, {decoded_count} decoded frames");
                break;
            };
            pkt_count += 1;
            if viewport_watcher.update() {
                let (w, h) = viewport_watcher.peek();
                decoder.set_viewport(*w, *h);
            }
            let t = Instant::now();
            decoder
                .push_packet(packet)
                .context("failed to push packet")?;
            trace!(t=?t.elapsed(), "videodec: push_packet");
            while let Some(frame) = decoder.pop_frame().context("failed to pop frame")? {
                decoded_count += 1;
                if decoded_count <= 3 || decoded_count % 100 == 0 {
                    let img = frame.img();
                    info!(
                        "videodec: decoded frame #{decoded_count} (from {pkt_count} packets), {}x{}, elapsed={:?}",
                        img.width(), img.height(), t.elapsed()
                    );
                }
                if output_tx.blocking_send(frame).is_err() {
                    warn!("videodec: output channel closed at decoded frame #{decoded_count}");
                    break;
                }
                trace!(t=?t.elapsed(), "videodec: tx");
            }
            if pkt_count <= 3 || pkt_count % 100 == 0 {
                info!("videodec: processed packet #{pkt_count}, total decoded frames: {decoded_count}");
            }
        }
        Ok(())
    }
}

async fn forward_frames(mut track: hang::TrackConsumer, sender: mpsc::Sender<hang::Frame>) {
    let mut count: u64 = 0;
    loop {
        let frame = track.read_frame().await;
        match frame {
            Ok(Some(frame)) => {
                count += 1;
                if count <= 3 || count % 100 == 0 {
                    info!(
                        "forward_frames: #{count} payload_bytes={} keyframe={}",
                        frame.payload.num_bytes(),
                        frame.keyframe
                    );
                }
                if sender.send(frame).await.is_err() {
                    break;
                }
            }
            Ok(None) => {
                info!("forward_frames: track ended after {count} frames");
                break;
            }
            Err(err) => {
                warn!("forward_frames: failed to read frame after {count}: {err:?}");
                break;
            }
        }
    }
}

#[cfg(any(feature = "video", feature = "video-ios"))]
pub struct AvRemoteTrack {
    pub broadcast: SubscribeBroadcast,
    pub video: Option<WatchTrack>,
    pub audio: Option<AudioTrack>,
}

#[cfg(any(feature = "video", feature = "video-ios"))]
impl AvRemoteTrack {
    pub fn new<D: Decoders>(
        broadcast: SubscribeBroadcast,
        audio_out: impl AudioSink,
        playback_config: PlaybackConfig,
    ) -> Result<Self> {
        let audio = broadcast
            .listen_with::<D::Audio>(playback_config.quality, audio_out)
            .inspect_err(|err| tracing::warn!("no audio track: {err}"))
            .ok();
        let video = broadcast
            .watch_with::<D::Video>(&playback_config.decode_config, playback_config.quality)
            .inspect_err(|err| tracing::warn!("no video track: {err}"))
            .ok();
        Ok(Self {
            broadcast,
            audio,
            video,
        })
    }
}
