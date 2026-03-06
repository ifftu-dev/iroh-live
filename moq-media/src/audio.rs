use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use firewheel::{
    FirewheelConfig, FirewheelContext,
    cpal::{CpalConfig, CpalEnumerator, CpalInputConfig, CpalOutputConfig},
    channel_config::{ChannelConfig, ChannelCount, NonZeroChannelCount},
    dsp::volume::{DEFAULT_DB_EPSILON, DbMeterNormalizer},
    graph::PortIdx,
    node::NodeID,
    nodes::{
        peak_meter::{PeakMeterNode, PeakMeterSmoother, PeakMeterState},
        stream::{
            ResamplingChannelConfig,
            reader::{StreamReaderConfig, StreamReaderNode, StreamReaderState},
            writer::{StreamWriterConfig, StreamWriterNode, StreamWriterState},
        },
    },
};
pub use firewheel::cpal::DeviceInfo;
pub use firewheel::cpal::cpal::DeviceId;
use tokio::sync::{mpsc, mpsc::error::TryRecvError, oneshot};
use tracing::{debug, error, info, trace, warn};

#[cfg(feature = "aec")]
use self::aec::{AecCaptureNode, AecProcessor, AecProcessorConfig, AecRenderNode};
use crate::{
    av::{AudioFormat, AudioSink, AudioSinkHandle, AudioSource},
    util::spawn_thread,
};

#[cfg(feature = "aec")]
mod aec;

type StreamWriterHandle = Arc<Mutex<StreamWriterState>>;
type StreamReaderHandle = Arc<Mutex<StreamReaderState>>;

#[derive(Debug, Clone)]
pub struct AudioBackend {
    tx: mpsc::Sender<DriverMessage>,
}

impl AudioBackend {
    pub fn new() -> Self {
        Self::new_with_devices(None, None)
    }

    /// Create an AudioBackend with specific input/output device IDs.
    ///
    /// Pass `None` for either to use the system default.
    /// Note: changing devices requires creating a new AudioBackend
    /// (the entire FirewheelContext is tied to one input + one output device).
    pub fn new_with_devices(
        input_device_id: Option<DeviceId>,
        output_device_id: Option<DeviceId>,
    ) -> Self {
        let (tx, rx) = mpsc::channel(32);
        let _handle = spawn_thread("audiodriver", move || {
            AudioDriver::new_with_devices(rx, input_device_id, output_device_id).run()
        });
        Self { tx }
    }

    /// List available audio input devices.
    pub fn list_input_devices() -> Vec<DeviceInfo> {
        CpalEnumerator.default_host().input_devices()
    }

    /// List available audio output devices.
    pub fn list_output_devices() -> Vec<DeviceInfo> {
        CpalEnumerator.default_host().output_devices()
    }

    pub async fn default_input(&self) -> Result<InputStream> {
        self.input(AudioFormat::mono_48k()).await
    }

    pub async fn input(&self, format: AudioFormat) -> Result<InputStream> {
        let (reply, reply_rx) = oneshot::channel();
        self.tx
            .send(DriverMessage::InputStream { format, reply })
            .await?;
        let (handle, peaks) = reply_rx.await??;
        Ok(InputStream {
            handle,
            format,
            peaks,
            normalizer: DbMeterNormalizer::new(-60., 0., -20.),
        })
    }

    pub async fn default_output(&self) -> Result<OutputStream> {
        self.output(AudioFormat::stereo_48k()).await
    }

    pub async fn output(&self, format: AudioFormat) -> Result<OutputStream> {
        let (reply, reply_rx) = oneshot::channel();
        self.tx
            .send(DriverMessage::OutputStream { format, reply })
            .await?;
        let handle = reply_rx.await??;
        Ok(handle)
    }
}

#[derive(Clone)]
pub struct OutputStream {
    handle: StreamWriterHandle,
    paused: Arc<AtomicBool>,
    peaks: Arc<Mutex<PeakMeterSmoother<2>>>,
    normalizer: DbMeterNormalizer,
}

impl AudioSinkHandle for OutputStream {
    fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }
    fn pause(&self) {
        self.paused.store(true, Ordering::Relaxed);
        self.handle.lock().expect("poisoned").pause_stream();
    }

    fn resume(&self) {
        self.paused.store(false, Ordering::Relaxed);
        self.handle.lock().expect("poisoned").resume();
    }

    fn toggle_pause(&self) {
        let was_paused = self.paused.fetch_xor(true, Ordering::Relaxed);
        if was_paused {
            self.handle.lock().expect("poisoned").resume();
        } else {
            self.handle.lock().expect("poisoned").pause_stream();
        }
    }

    fn smoothed_peak_normalized(&self) -> Option<f32> {
        Some(
            self.peaks
                .lock()
                .expect("poisoned")
                .smoothed_peaks_normalized_mono(&self.normalizer),
        )
    }
}

impl AudioSink for OutputStream {
    fn handle(&self) -> Box<dyn AudioSinkHandle> {
        Box::new(self.clone())
    }

    fn format(&self) -> Result<AudioFormat> {
        let info = self.handle.lock().expect("poisoned");
        let sample_rate = info
            .sample_rate()
            .context("output stream misses sample rate")?
            .get();
        let channel_count = info.num_channels().get().get();
        Ok(AudioFormat {
            sample_rate,
            channel_count,
        })
    }

    fn push_samples(&mut self, samples: &[f32]) -> Result<()> {
        let mut handle = self.handle.lock().unwrap();

        // If this happens excessively in Release mode, you may want to consider
        // increasing [`StreamWriterConfig::channel_config.latency_seconds`].
        if handle.underflow_occurred() {
            warn!("Underflow occured in stream writer node!");
        }

        // If this happens excessively in Release mode, you may want to consider
        // increasing [`StreamWriterConfig::channel_config.capacity_seconds`]. For
        // example, if you are streaming data from a network, you may want to
        // increase the capacity to several seconds.
        if handle.overflow_occurred() {
            warn!("Overflow occured in stream writer node!");
        }

        // Wait until the node's processor is ready to receive data.
        if handle.is_ready() {
            // let expected_bytes =
            //     frame.samples() * frame.channels() as usize * core::mem::size_of::<f32>();
            // let cpal_sample_data: &[f32] = bytemuck::cast_slice(&frame.data(0)[..expected_bytes]);
            handle.push_interleaved(samples);
            trace!("pushed samples {}", samples.len());
        } else {
            warn!("output handle is inactive")
        }
        Ok(())
    }
}

impl OutputStream {
    #[allow(unused)]
    pub fn is_active(&self) -> bool {
        self.handle.lock().expect("poisoned").is_active()
    }
}

/// A simple AudioSource that reads from the default microphone via Firewheel.
///
/// Includes a peak meter so the application can display a mic activity
/// indicator (VU meter) without modifying the audio samples.
#[derive(Clone)]
pub struct InputStream {
    handle: StreamReaderHandle,
    format: AudioFormat,
    peaks: Arc<Mutex<PeakMeterSmoother<1>>>,
    normalizer: DbMeterNormalizer,
}

impl InputStream {
    /// Get the smoothed peak level of the microphone input, normalized
    /// to a 0.0–1.0 range suitable for driving a VU meter UI.
    pub fn smoothed_peak_normalized(&self) -> f32 {
        self.peaks
            .lock()
            .expect("poisoned")
            .smoothed_peaks_normalized_mono(&self.normalizer)
    }
}

impl AudioSource for InputStream {
    fn cloned_boxed(&self) -> Box<dyn AudioSource> {
        Box::new(self.clone())
    }

    fn format(&self) -> AudioFormat {
        self.format
    }

    fn pop_samples(&mut self, buf: &mut [f32]) -> Result<Option<usize>> {
        use firewheel::nodes::stream::ReadStatus;
        let mut handle = self.handle.lock().expect("poisoned");
        match handle.read_interleaved(buf) {
            Some(ReadStatus::Ok) => Ok(Some(buf.len())),
            Some(ReadStatus::InputNotReady) => {
                tracing::warn!("audio input not ready");
                // Maintain pacing; still return a frame-sized buffer
                Ok(Some(buf.len()))
            }
            Some(ReadStatus::UnderflowOccurred { num_frames_read }) => {
                tracing::warn!(
                    "audio input underflow: {} frames missing",
                    buf.len() - num_frames_read
                );
                Ok(Some(buf.len()))
            }
            Some(ReadStatus::OverflowCorrected {
                num_frames_discarded,
            }) => {
                tracing::warn!("audio input overflow: {num_frames_discarded} frames discarded");
                Ok(Some(buf.len()))
            }
            None => {
                tracing::warn!("audio input stream is inactive");
                Ok(None)
            }
        }
    }
}

#[derive(derive_more::Debug)]
enum DriverMessage {
    OutputStream {
        format: AudioFormat,
        #[debug("Sender")]
        reply: oneshot::Sender<Result<OutputStream>>,
    },
    InputStream {
        format: AudioFormat,
        #[debug("Sender")]
        reply: oneshot::Sender<Result<(StreamReaderHandle, Arc<Mutex<PeakMeterSmoother<1>>>)>>,
    },
}

struct AudioDriver {
    cx: FirewheelContext,
    rx: mpsc::Receiver<DriverMessage>,
    #[cfg(feature = "aec")]
    aec_processor: AecProcessor,
    /// The node that sits between stream writers (output) and graph_out.
    /// With AEC: this is the AecRenderNode. Without AEC: this is graph_out directly.
    output_anchor: NodeID,
    /// The node that sits between graph_in and stream readers (input).
    /// With AEC: this is the AecCaptureNode. Without AEC: this is graph_in directly.
    input_anchor: NodeID,
    peak_meters: HashMap<NodeID, Arc<Mutex<PeakMeterSmoother<2>>>>,
    mono_peak_meters: HashMap<NodeID, Arc<Mutex<PeakMeterSmoother<1>>>>,
}

impl AudioDriver {
    fn new(rx: mpsc::Receiver<DriverMessage>) -> Self {
        Self::new_with_devices(rx, None, None)
    }

    fn new_with_devices(
        rx: mpsc::Receiver<DriverMessage>,
        input_device_id: Option<DeviceId>,
        output_device_id: Option<DeviceId>,
    ) -> Self {
        let config = FirewheelConfig {
            num_graph_inputs: ChannelCount::new(1).unwrap(),
            ..Default::default()
        };
        let mut cx = FirewheelContext::new(config);
        let config = CpalConfig {
            output: CpalOutputConfig {
                device_id: output_device_id,
                ..Default::default()
            },
            input: Some(CpalInputConfig {
                device_id: input_device_id,
                fail_on_no_input: true,
                ..Default::default()
            }),
        };
        cx.start_stream(config).unwrap();
        info!(
            "audio graph in: {:?}",
            cx.node_info(cx.graph_in_node_id()).map(|x| &x.info)
        );
        info!(
            "audio graph out: {:?}",
            cx.node_info(cx.graph_out_node_id()).map(|x| &x.info)
        );

        cx.set_graph_channel_config(ChannelConfig {
            num_inputs: ChannelCount::new(2).unwrap(),
            num_outputs: ChannelCount::new(2).unwrap(),
        });

        #[cfg(feature = "aec")]
        let (output_anchor, input_anchor, aec_processor) = {
            let aec_processor = AecProcessor::new(AecProcessorConfig::stereo_in_out(), true)
                .expect("failed to initialize AEC processor");
            let aec_render_node =
                cx.add_node(AecRenderNode::default(), Some(aec_processor.clone()));
            let aec_capture_node =
                cx.add_node(AecCaptureNode::default(), Some(aec_processor.clone()));

            let layout = &[(0, 0), (1, 1)];
            cx.connect(cx.graph_in_node_id(), aec_capture_node, layout, true)
                .unwrap();
            cx.connect(aec_render_node, cx.graph_out_node_id(), layout, true)
                .unwrap();

            info!("AEC enabled: inserted capture and render nodes into audio graph");
            (aec_render_node, aec_capture_node, aec_processor)
        };

        #[cfg(not(feature = "aec"))]
        let (output_anchor, input_anchor) = {
            // Without AEC, output streams connect directly to graph_out,
            // and input streams read directly from graph_in.
            info!("AEC disabled: using direct graph connections (no echo cancellation)");
            (cx.graph_out_node_id(), cx.graph_in_node_id())
        };

        Self {
            cx,
            rx,
            #[cfg(feature = "aec")]
            aec_processor,
            output_anchor,
            input_anchor,
            peak_meters: Default::default(),
            mono_peak_meters: Default::default(),
        }
    }

    fn run(&mut self) {
        const INTERVAL: Duration = Duration::from_millis(10);
        const PEAK_UPDATE_INTERVAL: Duration = Duration::from_millis(40);
        #[cfg(feature = "aec")]
        let mut last_delay: f64 = 0.;
        let mut last_peak_update = Instant::now();

        loop {
            let tick = Instant::now();
            if self.drain_messages().is_err() {
                info!("closing audio driver: message channel closed");
                break;
            }

            if let Err(e) = self.cx.update() {
                error!("audio backend error: {:?}", &e);

                // if let UpdateError::StreamStoppedUnexpectedly(_) = e {
                //     // Notify the stream node handles that the output stream has stopped.
                //     // This will automatically stop any active streams on the nodes.
                //     cx.node_state_mut::<StreamWriterState>(stream_writer_id)
                //         .unwrap()
                //         .stop_stream();
                //     cx.node_state_mut::<StreamReaderState>(stream_reader_id)
                //         .unwrap()
                //         .stop_stream();

                //     // The stream has stopped unexpectedly (i.e the user has
                //     // unplugged their headphones.)
                //     //
                //     // Typically you should start a new stream as soon as
                //     // possible to resume processing (event if it's a dummy
                //     // output device).
                //     //
                //     // In this example we just quit the application.
                //     break;
                // }
            }

            #[cfg(feature = "aec")]
            if let Some(info) = self.cx.stream_info() {
                let delay = info.input_to_output_latency_seconds;
                if (last_delay - delay).abs() > (1. / 1000.) {
                    let delay_ms = (delay * 1000.) as u32;
                    info!("update processor delay to {delay_ms}ms");
                    self.aec_processor.set_stream_delay(delay_ms);
                    last_delay = delay;
                }
            }

            // Update peak meters
            let delta = last_peak_update.elapsed();
            if delta > PEAK_UPDATE_INTERVAL {
                for (id, smoother) in self.peak_meters.iter_mut() {
                    smoother.lock().expect("poisoned").update(
                        self.cx
                            .node_state::<PeakMeterState<2>>(*id)
                            .unwrap()
                            .peak_gain_db(DEFAULT_DB_EPSILON),
                        delta.as_secs_f32(),
                    );
                }
                for (id, smoother) in self.mono_peak_meters.iter_mut() {
                    smoother.lock().expect("poisoned").update(
                        self.cx
                            .node_state::<PeakMeterState<1>>(*id)
                            .unwrap()
                            .peak_gain_db(DEFAULT_DB_EPSILON),
                        delta.as_secs_f32(),
                    );
                }
                last_peak_update = Instant::now();
            }

            std::thread::sleep(INTERVAL.saturating_sub(tick.elapsed()));
        }
    }

    fn drain_messages(&mut self) -> Result<(), ()> {
        loop {
            match self.rx.try_recv() {
                Err(TryRecvError::Disconnected) => {
                    info!("stopping audio thread: backend handle dropped");
                    break Err(());
                }
                Err(TryRecvError::Empty) => {
                    break Ok(());
                }
                Ok(message) => self.handle_message(message),
            }
        }
    }

    fn handle_message(&mut self, message: DriverMessage) {
        debug!("handle {message:?}");
        match message {
            DriverMessage::OutputStream { format, reply } => {
                let res = self
                    .output_stream(format)
                    .inspect_err(|err| warn!("failed to create audio output stream: {err:#}"));
                reply.send(res).ok();
            }
            DriverMessage::InputStream { format, reply } => {
                let res = self
                    .input_stream(format)
                    .inspect_err(|err| warn!("failed to create audio input stream: {err:#}"));
                reply.send(res).ok();
            }
        }
    }

    fn output_stream(&mut self, format: AudioFormat) -> Result<OutputStream> {
        let channel_count = format.channel_count;
        let sample_rate = format.sample_rate;
        // setup stream
        let stream_writer_id = self.cx.add_node(
            StreamWriterNode,
            Some(StreamWriterConfig {
                channels: NonZeroChannelCount::new(channel_count)
                    .context("channel count may not be zero")?,
                ..Default::default()
            }),
        );
        let graph_out = self.output_anchor;
        // let graph_out_info = self
        //     .cx
        //     .node_info(graph_out)
        //     .context("missing audio output node")?;

        let peak_meter_node = PeakMeterNode::<2> { enabled: true };
        let peak_meter_id = self.cx.add_node(peak_meter_node.clone(), None);
        let peak_meter_smoother =
            Arc::new(Mutex::new(PeakMeterSmoother::<2>::new(Default::default())));
        self.peak_meters
            .insert(peak_meter_id, peak_meter_smoother.clone());
        self.cx
            .connect(peak_meter_id, graph_out, &[(0, 0), (1, 1)], true)
            .unwrap();

        let layout: &[(PortIdx, PortIdx)] = match channel_count {
            0 => anyhow::bail!("audio stream has no channels"),
            1 => &[(0, 0), (0, 1)],
            _ => &[(0, 0), (1, 1)],
        };
        self.cx
            .connect(stream_writer_id, peak_meter_id, layout, false)
            .unwrap();
        let output_stream_sample_rate = self.cx.stream_info().unwrap().sample_rate;
        let event = self
            .cx
            .node_state_mut::<StreamWriterState>(stream_writer_id)
            .unwrap()
            .start_stream(
                sample_rate.try_into().unwrap(),
                output_stream_sample_rate,
                ResamplingChannelConfig {
                    capacity_seconds: 3.,
                    ..Default::default()
                },
            )
            .unwrap();
        info!("started output stream");
        self.cx.queue_event_for(stream_writer_id, event.into());
        // Wrap the handles in an `Arc<Mutex<T>>>` so that we can send them to other threads.
        let handle = self
            .cx
            .node_state::<StreamWriterState>(stream_writer_id)
            .unwrap()
            .handle();
        Ok(OutputStream {
            handle: Arc::new(handle),
            paused: Arc::new(AtomicBool::new(false)),
            peaks: peak_meter_smoother,
            normalizer: DbMeterNormalizer::new(-60., 0., -20.),
        })
    }

    fn input_stream(
        &mut self,
        format: AudioFormat,
    ) -> Result<(StreamReaderHandle, Arc<Mutex<PeakMeterSmoother<1>>>)> {
        let sample_rate = format.sample_rate;
        let channel_count = format.channel_count;
        // Setup stream reader node
        let stream_reader_id = self.cx.add_node(
            StreamReaderNode,
            Some(StreamReaderConfig {
                channels: NonZeroChannelCount::new(channel_count)
                    .context("channel count may not be zero")?,
                ..Default::default()
            }),
        );

        // Insert a mono peak meter between the AEC capture node and the stream reader.
        let peak_meter_node = PeakMeterNode::<1> { enabled: true };
        let peak_meter_id = self.cx.add_node(peak_meter_node, None);
        let peak_meter_smoother =
            Arc::new(Mutex::new(PeakMeterSmoother::<1>::new(Default::default())));
        self.mono_peak_meters
            .insert(peak_meter_id, peak_meter_smoother.clone());

        let graph_in_node_id = self.input_anchor;
        let num_capture_outputs = self
            .cx
            .node_info(graph_in_node_id)
            .context("missing audio input node")?
            .info
            .channel_config
            .num_outputs
            .get();

        // Connect: aec_capture → peak_meter (mono)
        if num_capture_outputs == 0 {
            anyhow::bail!("audio input has no channels");
        }
        // Always take first channel for mono peak meter
        self.cx
            .connect(graph_in_node_id, peak_meter_id, &[(0, 0)], false)
            .unwrap();

        // Connect: peak_meter (mono) → stream_reader
        let meter_to_reader: &[(PortIdx, PortIdx)] = match channel_count {
            0 => anyhow::bail!("audio stream has no channels"),
            1 => &[(0, 0)],
            _ => &[(0, 0), (0, 1)], // duplicate mono to stereo if needed
        };
        self.cx
            .connect(peak_meter_id, stream_reader_id, meter_to_reader, false)
            .unwrap();

        let input_stream_sample_rate = self.cx.stream_info().unwrap().sample_rate;
        let event = self
            .cx
            .node_state_mut::<StreamReaderState>(stream_reader_id)
            .unwrap()
            .start_stream(
                sample_rate.try_into().unwrap(),
                input_stream_sample_rate,
                ResamplingChannelConfig {
                    capacity_seconds: 3.0,
                    ..Default::default()
                },
            )
            .unwrap();
        self.cx.queue_event_for(stream_reader_id, event.into());

        let handle = self
            .cx
            .node_state::<StreamReaderState>(stream_reader_id)
            .unwrap()
            .handle();
        Ok((Arc::new(handle), peak_meter_smoother))
    }
}
