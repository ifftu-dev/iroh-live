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

use std::time::Duration;

use iroh::{Endpoint, EndpointAddr};
use iroh_moq::{Moq, MoqProtocolHandler, MoqSession};
use moq_lite::BroadcastProducer;
use moq_media::subscribe::SubscribeBroadcast;
#[cfg(feature = "video")]
use moq_media::{
    av::{AudioSink, Decoders, PlaybackConfig},
    subscribe::AvRemoteTrack,
};
use n0_error::Result;
use tracing::{info, warn};

#[derive(Clone)]
pub struct Live {
    pub moq: Moq,
}

impl Live {
    pub fn new(endpoint: Endpoint) -> Self {
        Self {
            moq: Moq::new(endpoint),
        }
    }

    pub async fn connect(&self, remote: impl Into<EndpointAddr>) -> Result<MoqSession> {
        self.moq.connect(remote).await
    }

    pub async fn connect_and_subscribe(
        &self,
        remote: impl Into<EndpointAddr>,
        broadcast_name: &str,
    ) -> Result<(MoqSession, SubscribeBroadcast)> {
        let remote = remote.into();
        let mut session = self.connect(remote.clone()).await?;
        info!(id=%session.conn().remote_id(), "new peer connected");

        let broadcast = match tokio::time::timeout(
            Duration::from_secs(5),
            session.subscribe(broadcast_name),
        )
        .await
        {
            Ok(Ok(broadcast)) => broadcast,
            Ok(Err(err)) => {
                session.close(0, b"subscribe failed");
                return Err(err.into());
            }
            Err(_) => {
                warn!(
                    id = %session.conn().remote_id(),
                    broadcast = %broadcast_name,
                    "subscribe timed out on cached session, closing and retrying with a fresh connection"
                );
                session.close(0, b"subscribe timeout");

                let mut session = self.connect(remote).await?;
                info!(
                    id = %session.conn().remote_id(),
                    broadcast = %broadcast_name,
                    "reconnected peer after subscribe timeout"
                );
                let broadcast =
                    tokio::time::timeout(Duration::from_secs(5), session.subscribe(broadcast_name))
                        .await
                        .map_err(|_| {
                            session.close(0, b"subscribe retry timeout");
                            n0_error::anyerr!(
                                "subscribe to '{broadcast_name}' timed out after reconnect"
                            )
                        })?
                        .map_err(|err| {
                            session.close(0, b"subscribe retry failed");
                            err
                        })?;
                let broadcast =
                    SubscribeBroadcast::new(broadcast_name.to_string(), broadcast).await?;
                return Ok((session, broadcast));
            }
        };

        let broadcast = SubscribeBroadcast::new(broadcast_name.to_string(), broadcast).await?;
        Ok((session, broadcast))
    }

    #[cfg(feature = "video")]
    pub async fn watch_and_listen<D: Decoders>(
        &self,
        remote: impl Into<EndpointAddr>,
        broadcast_name: &str,
        audio_out: impl AudioSink,
        config: PlaybackConfig,
    ) -> Result<(MoqSession, AvRemoteTrack)> {
        let (session, broadcast) = self.connect_and_subscribe(remote, &broadcast_name).await?;
        let track = broadcast.watch_and_listen::<D>(audio_out, config)?;
        Ok((session, track))
    }

    pub fn protocol_handler(&self) -> MoqProtocolHandler {
        self.moq.protocol_handler()
    }

    pub async fn publish(&self, name: impl ToString, producer: BroadcastProducer) -> Result<()> {
        self.moq.publish(name, producer).await
    }

    pub fn shutdown(&self) {
        self.moq.shutdown();
    }
}
