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


use std::sync::{Arc, Mutex};

use moq_lite::BroadcastProducer;
use moq_media::{
    audio::AudioBackend,
    av::AudioPreset,
    publish::{AudioRenditions, PublishBroadcast},
};
#[cfg(feature = "video")]
use moq_media::{
    av::VideoPreset,
    capture::{CameraCapturer, ScreenCapturer},
    ffmpeg::{H264Encoder, OpusEncoder},
    publish::VideoRenditions,
};
#[cfg(not(feature = "video"))]
use moq_media::opus::PureOpusEncoder;
use n0_error::{AnyError, Result};
use tracing::{info, warn};

use crate::rooms::RoomHandle;

#[derive(Debug, strum::Display, strum::EnumString)]
#[strum(serialize_all = "lowercase")]
enum Broadcasts {
    Camera,
    Screen,
}

#[derive(Debug)]
pub enum StreamKind {
    Camera,
    Screen,
    Microphone,
}

#[derive(Default, Clone, Debug)]
pub struct PublishOpts {
    pub camera: bool,
    pub screen: bool,
    pub audio: bool,
}

/// Manager for publish broadcasts in a room
///
/// Synchronous version which spawns all async ops on new tokio tasks. Panics if methods are
/// not called in the context of a tokio runtime.
///
/// Why does this have sync methods? In UI land it is so much easier for the operations to be sync,
/// so this just spawns all async ops on tokio threads. Not yet sure about where this should evolve to
/// but this kept me moving for now.
pub struct RoomPublisherSync {
    audio_ctx: AudioBackend,
    room: RoomHandle,
    camera: Option<Arc<Mutex<PublishBroadcast>>>,
    screen: Option<Arc<Mutex<PublishBroadcast>>>,
    state: PublishOpts,
}

impl RoomPublisherSync {
    pub fn new(room: RoomHandle, audio_ctx: AudioBackend) -> Self {
        Self {
            room,
            audio_ctx,
            camera: None,
            screen: None,
            state: Default::default(),
        }
    }

    pub fn set_state(&mut self, state: &PublishOpts) -> Result<(), Vec<(StreamKind, AnyError)>> {
        info!(new=?state, old=?self.state, "set publish state");
        let errors = [
            self.set_audio(state.audio)
                .err()
                .map(|e| (StreamKind::Microphone, e)),
            self.set_camera(state.camera)
                .err()
                .map(|e| (StreamKind::Camera, e)),
            self.set_screen(state.screen)
                .err()
                .map(|e| (StreamKind::Screen, e)),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn state(&self) -> &PublishOpts {
        &self.state
    }

    pub fn camera(&self) -> bool {
        self.state.camera
    }

    pub fn camera_broadcast(&self) -> Option<Arc<Mutex<PublishBroadcast>>> {
        self.camera.clone()
    }

    pub fn screen_broadcast(&self) -> Option<Arc<Mutex<PublishBroadcast>>> {
        self.screen.clone()
    }

    #[cfg(feature = "video")]
    pub fn set_camera(&mut self, enable: bool) -> Result<()> {
        if self.state.camera != enable {
            if enable {
                let camera = CameraCapturer::new()?;
                let renditions = VideoRenditions::new::<H264Encoder>(camera, VideoPreset::all());
                self.ensure_camera();
                self.camera
                    .as_ref()
                    .unwrap()
                    .lock()
                    .unwrap()
                    .set_video(Some(renditions))?;
            } else if let Some(camera) = self.camera.as_ref() {
                camera.lock().unwrap().set_video(None)?;
            }
            self.state.camera = enable;
        }
        Ok(())
    }

    #[cfg(not(feature = "video"))]
    pub fn set_camera(&mut self, _enable: bool) -> Result<()> {
        Err(n0_error::anyerr!("Camera is not available without the video feature"))
    }

    pub fn screen(&self) -> bool {
        self.state.screen
    }

    fn ensure_camera(&mut self) {
        if self.camera.is_none() {
            let broadcast = PublishBroadcast::new();
            self.publish(Broadcasts::Camera, broadcast.producer());
            self.camera = Some(Arc::new(Mutex::new(broadcast)));
        };
    }

    fn publish(&self, name: Broadcasts, producer: BroadcastProducer) {
        let room = self.room.clone();
        tokio::spawn(async move {
            if let Err(err) = room.publish(name, producer).await {
                warn!("publish to room failed: {err:#}");
            }
        });
    }

    #[cfg(feature = "video")]
    pub fn set_screen(&mut self, enable: bool) -> Result<()> {
        if self.state.screen != enable {
            if enable {
                if self.screen.is_none() {
                    let broadcast = PublishBroadcast::new();
                    self.publish(Broadcasts::Screen, broadcast.producer());
                    self.screen = Some(Arc::new(Mutex::new(broadcast)));
                };

                let screen = ScreenCapturer::new()?;
                let renditions = VideoRenditions::new::<H264Encoder>(screen, VideoPreset::all());
                self.screen
                    .as_mut()
                    .unwrap()
                    .lock()
                    .unwrap()
                    .set_video(Some(renditions))?;
            } else {
                let _ = self.screen.take();
            }
            self.state.screen = enable;
        }
        Ok(())
    }

    #[cfg(not(feature = "video"))]
    pub fn set_screen(&mut self, _enable: bool) -> Result<()> {
        Err(n0_error::anyerr!("Screen sharing is not available without the video feature"))
    }

    pub fn audio(&self) -> bool {
        self.state.audio
    }

    pub fn set_audio(&mut self, enable: bool) -> Result<()> {
        if self.state.audio != enable {
            if enable {
                self.ensure_camera();
                let camera = self.camera.as_ref().unwrap().clone();
                let audio_ctx = self.audio_ctx.clone();
                tokio::spawn(async move {
                    let mic = match audio_ctx.default_input().await {
                        Err(err) => {
                            warn!("failed to open audio input: {err:#}");
                            return;
                        }
                        Ok(mic) => mic,
                    };
                    #[cfg(feature = "video")]
                    let renditions = AudioRenditions::new::<OpusEncoder>(mic, [AudioPreset::Hq]);
                    #[cfg(not(feature = "video"))]
                    let renditions = AudioRenditions::new::<PureOpusEncoder>(mic, [AudioPreset::Hq]);
                    if let Err(err) = camera.lock().unwrap().set_audio(Some(renditions)) {
                        warn!("failed to set audio: {err:#}");
                    }
                });
            } else if let Some(camera) = self.camera.as_mut() {
                camera.lock().unwrap().set_audio(None)?;
            }
            self.state.audio = enable;
        }
        Ok(())
    }
}
