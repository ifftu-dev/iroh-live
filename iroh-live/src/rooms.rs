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

use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use iroh::{Endpoint, EndpointAddr, EndpointId, SecretKey};
use iroh_gossip::Gossip;
use iroh_moq::MoqSession;
use iroh_smol_kv::{ExpiryConfig, Filter, SignedValue, Subscribe, SubscribeMode, WriteScope};
use moq_lite::BroadcastProducer;
use moq_media::subscribe::SubscribeBroadcast;
use n0_error::{Result, StdResultExt, anyerr};
use n0_future::FuturesUnordered;
use n0_future::{StreamExt, task::AbortOnDropHandle};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::{self, error::TryRecvError};
use tracing::{Instrument, debug, error_span, info, warn};

use crate::Live;

pub use self::publisher::{PublishOpts, RoomPublisherSync, StreamKind};
pub use self::ticket::RoomTicket;

type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send + Sync + 'static>>;

mod publisher;

pub struct Room {
    handle: RoomHandle,
    events: mpsc::Receiver<RoomEvent>,
}

pub type RoomEvents = mpsc::Receiver<RoomEvent>;

#[derive(Clone)]
pub struct RoomHandle {
    me: EndpointId,
    ticket: RoomTicket,
    tx: mpsc::Sender<ApiMessage>,
    _actor_handle: Arc<AbortOnDropHandle<()>>,
}

impl RoomHandle {
    pub fn ticket(&self) -> RoomTicket {
        let mut ticket = self.ticket.clone();
        ticket.bootstrap = vec![self.me];
        ticket
    }

    pub async fn publish(&self, name: impl ToString, producer: BroadcastProducer) -> Result<()> {
        self.tx
            .send(ApiMessage::Publish {
                name: name.to_string(),
                producer: producer,
            })
            .await
            .map_err(|_| anyerr!("room actor died"))
    }

    /// Force the room actor to remove a broadcast from active subscriptions and reconnect.
    /// Call this when you detect a subscription has died (e.g., frame bridge ended)
    /// but `broadcast.closed()` hasn't fired.
    pub async fn force_resubscribe(&self, remote: EndpointId, name: impl ToString) -> Result<()> {
        self.tx
            .send(ApiMessage::ForceResubscribe {
                remote,
                name: name.to_string(),
            })
            .await
            .map_err(|_| anyerr!("room actor died"))
    }
}

impl Room {
    pub async fn new(
        endpoint: &Endpoint,
        gossip: Gossip,
        live: Live,
        ticket: RoomTicket,
    ) -> Result<Self> {
        let endpoint_id = endpoint.id();
        let (actor_tx, actor_rx) = mpsc::channel(16);
        let (event_tx, event_rx) = mpsc::channel(16);

        let actor = Actor::new(
            endpoint.secret_key(),
            endpoint.clone(),
            live,
            event_tx,
            gossip,
            ticket.clone(),
        )
        .await?;
        let actor_task = tokio::task::spawn(
            async move { actor.run(actor_rx).await }
                .instrument(error_span!("RoomActor", id = ticket.topic_id.fmt_short())),
        );

        Ok(Self {
            handle: RoomHandle {
                ticket,
                me: endpoint_id,
                tx: actor_tx,
                _actor_handle: Arc::new(AbortOnDropHandle::new(actor_task)),
            },
            events: event_rx,
        })
    }

    pub async fn recv(&mut self) -> Result<RoomEvent> {
        self.events.recv().await.std_context("sender stopped")
    }

    pub fn try_recv(&mut self) -> Result<RoomEvent, TryRecvError> {
        self.events.try_recv()
    }

    pub fn ticket(&self) -> RoomTicket {
        self.handle.ticket()
    }

    pub fn split(self) -> (RoomEvents, RoomHandle) {
        (self.events, self.handle)
    }

    pub async fn publish(&self, name: impl ToString, producer: BroadcastProducer) -> Result<()> {
        self.handle.publish(name, producer).await
    }
}

enum ApiMessage {
    Publish {
        name: String,
        producer: BroadcastProducer,
    },
    ForceResubscribe {
        remote: EndpointId,
        name: String,
    },
}

pub enum RoomEvent {
    RemoteAnnounced {
        remote: EndpointId,
        broadcasts: Vec<String>,
    },
    RemoteConnected {
        session: MoqSession,
    },
    BroadcastSubscribed {
        session: MoqSession,
        broadcast: SubscribeBroadcast,
    },
}

const PEER_STATE_KEY: &[u8] = b"s";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PeerState {
    broadcasts: Vec<String>,
}

type KvEntry = (EndpointId, Bytes, SignedValue);

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, derive_more::Display)]
#[display("{}:{}", _0.fmt_short(), _1)]
struct BroadcastId(EndpointId, String);

struct Actor {
    me: EndpointId,
    endpoint: Endpoint,
    _gossip: Gossip,
    live: Live,
    active_subscribe: HashSet<BroadcastId>,
    active_publish: HashSet<String>,
    known_remote_broadcasts: HashMap<EndpointId, Vec<String>>,
    active_sessions: HashMap<BroadcastId, MoqSession>,
    peer_addrs: HashMap<EndpointId, EndpointAddr>,
    retry_counts: HashMap<BroadcastId, u32>,
    connecting:
        FuturesUnordered<BoxFuture<(BroadcastId, Result<(MoqSession, SubscribeBroadcast)>)>>,
    subscribe_closed: FuturesUnordered<BoxFuture<BroadcastId>>,
    retry_timers: FuturesUnordered<BoxFuture<BroadcastId>>,
    publish_closed: FuturesUnordered<BoxFuture<String>>,
    event_tx: mpsc::Sender<RoomEvent>,
    kv: iroh_smol_kv::Client,
    kv_writer: WriteScope,
}

impl Actor {
    async fn new(
        me: &SecretKey,
        endpoint: Endpoint,
        live: Live,
        event_tx: mpsc::Sender<RoomEvent>,
        gossip: Gossip,
        ticket: RoomTicket,
    ) -> Result<Self> {
        let topic = gossip
            .subscribe(ticket.topic_id, ticket.bootstrap.clone())
            .await?;
        let kv = iroh_smol_kv::Client::local(
            topic,
            iroh_smol_kv::Config {
                anti_entropy_interval: Duration::from_secs(60),
                fast_anti_entropy_interval: Duration::from_secs(1),
                expiry: Some(ExpiryConfig {
                    check_interval: Duration::from_secs(10),
                    horizon: Duration::from_secs(60 * 2),
                }),
            },
        );
        let kv_writer = kv.write(me.clone());
        Ok(Self {
            me: me.public(),
            endpoint,
            live,
            _gossip: gossip,
            active_subscribe: Default::default(),
            active_publish: Default::default(),
            known_remote_broadcasts: Default::default(),
            active_sessions: Default::default(),
            peer_addrs: Default::default(),
            retry_counts: Default::default(),
            connecting: Default::default(),
            subscribe_closed: Default::default(),
            retry_timers: Default::default(),
            publish_closed: Default::default(),
            event_tx,
            kv,
            kv_writer,
        })
    }

    pub async fn run(mut self, mut inbox: mpsc::Receiver<ApiMessage>) {
        let updates = self
            .kv
            .subscribe_with_opts(Subscribe {
                mode: SubscribeMode::Both,
                filter: Filter::ALL,
            })
            .stream();
        tokio::pin!(updates);

        // Periodic reconciliation: every 15s, check if any known remote broadcasts
        // are missing from active_subscribe and reconnect them.
        let mut reconcile_interval = tokio::time::interval(Duration::from_secs(15));
        reconcile_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        loop {
            tokio::select! {
                Some(update) = updates.next() => {
                    match update {
                        Err(err) => warn!("gossip kv update failed: {err:#}"),
                        Ok(update) => self.handle_gossip_update(update).await,
                    }
                }
                msg = inbox.recv() => {
                    match msg {
                        None => break,
                        Some(msg) => self.handle_api_message(msg).await
                    }
                }
                Some((id, res)) = self.connecting.next(), if !self.connecting.is_empty() => {
                    match res {
                        Ok((session, broadcast)) => {
                            let peer_id = session.remote_id();
                            let mut peer_addr = EndpointAddr::from(peer_id);
                            let our_addr = self.endpoint.addr();
                            for a in &our_addr.addrs {
                                if a.is_relay() {
                                    peer_addr.addrs.insert(a.clone());
                                }
                            }
                            self.peer_addrs.insert(peer_id, peer_addr);
                            self.active_sessions.insert(id.clone(), session.clone());
                            self.retry_counts.remove(&id);
                            let closed_fut = broadcast.closed();
                            self.event_tx.send(RoomEvent::BroadcastSubscribed { session, broadcast }).await.ok();
                            self.subscribe_closed.push(Box::pin(async move {
                                closed_fut.await;
                                id
                            }))
                        }
                        Err(err) => {
                            self.active_subscribe.remove(&id);
                            warn!("Subscribing to broadcast {id} failed: {err:#}");
                            self.schedule_retry(id);
                        }
                    }
                }
                Some(id) = self.subscribe_closed.next(), if !self.subscribe_closed.is_empty() => {
                    self.active_subscribe.remove(&id);
                    // Keep the MoQ session alive across subscription repair.
                    // The session is shared bidirectionally (pub + sub), so
                    // dropping it here can kill the reverse publishing tracks
                    // and collapse the whole tutoring connection.
                    self.schedule_retry(id);
                }
                Some(id) = self.retry_timers.next(), if !self.retry_timers.is_empty() => {
                    self.handle_retry(id);
                }
                _ = reconcile_interval.tick() => {
                    self.reconcile_subscriptions();
                }
                Some(name) = self.publish_closed.next(), if !self.publish_closed.is_empty() => {
                    self.active_publish.remove(&name);
                    self.update_kv().await;
                }
            }
        }
    }

    async fn handle_api_message(&mut self, msg: ApiMessage) {
        match msg {
            ApiMessage::Publish { name, producer } => {
                let closed = producer.consume().closed();
                self.live.publish(name.clone(), producer).await.ok();
                self.active_publish.insert(name.clone());
                self.publish_closed.push(Box::pin(async move {
                    closed.await;
                    name
                }));
                self.update_kv().await;
            }
            ApiMessage::ForceResubscribe { remote, name } => {
                let id = BroadcastId(remote, name.clone());
                info!("force_resubscribe: {id} — removing from active and reconnecting");
                self.active_subscribe.remove(&id);
                self.retry_counts.remove(&id);
                // Don't close the session — it's shared bidirectionally.
                // handle_connect will reuse it if healthy or create a new one.
                self.active_sessions.remove(&id);
                let still_known = self
                    .known_remote_broadcasts
                    .get(&remote)
                    .map(|names| names.contains(&name))
                    .unwrap_or(false);
                if still_known {
                    self.active_subscribe.insert(id.clone());
                    self.start_connect_and_subscribe(id, remote, name);
                } else {
                    info!(
                        "force_resubscribe: {id} — peer no longer in known_remote_broadcasts, skipping"
                    );
                }
            }
        }
    }

    async fn handle_gossip_update(&mut self, entry: KvEntry) {
        let (remote, key, value) = entry;
        if remote == self.me || &key != PEER_STATE_KEY {
            return;
        }
        let Ok(value) = postcard::from_bytes::<PeerState>(&value.value) else {
            return;
        };
        let PeerState { broadcasts } = value;
        // Track known remote broadcasts for reconciliation.
        self.known_remote_broadcasts
            .insert(remote, broadcasts.clone());
        for name in broadcasts.clone() {
            let id = BroadcastId(remote, name.clone());
            if !self.active_subscribe.insert(id.clone()) {
                continue;
            }
            self.start_connect_and_subscribe(id, remote, name);
        }
        self.event_tx
            .send(RoomEvent::RemoteAnnounced { remote, broadcasts })
            .await
            .ok();
    }

    /// Attempt to connect and subscribe to a broadcast.
    fn start_connect_and_subscribe(&mut self, id: BroadcastId, remote: EndpointId, name: String) {
        let live = self.live.clone();
        let addr = self.peer_addrs.get(&remote).cloned().unwrap_or_else(|| {
            let mut addr = EndpointAddr::from(remote);
            let our_addr = self.endpoint.addr();
            for a in &our_addr.addrs {
                if a.is_relay() {
                    addr.addrs.insert(a.clone());
                }
            }
            addr
        });
        self.connecting.push(Box::pin(async move {
            info!(
                remote = %remote.fmt_short(),
                name = %name,
                addrs = addr.addrs.len(),
                "room: starting connect_and_subscribe"
            );
            let result = tokio::time::timeout(
                Duration::from_secs(10),
                live.connect_and_subscribe(addr, &name),
            )
            .await;
            let session = match result {
                Ok(inner) => {
                    match &inner {
                        Ok(_) => info!(
                            remote = %remote.fmt_short(),
                            name = %name,
                            "room: connect_and_subscribe succeeded"
                        ),
                        Err(e) => warn!(
                            remote = %remote.fmt_short(),
                            name = %name,
                            error = %e,
                            "room: connect_and_subscribe failed"
                        ),
                    }
                    inner
                }
                Err(_) => {
                    warn!(
                        remote = %remote.fmt_short(),
                        name = %name,
                        "room: connect_and_subscribe timed out after 10s"
                    );
                    Err(n0_error::anyerr!(
                        "connect_and_subscribe timed out after 10s"
                    ))
                }
            };
            (id, session)
        }));
    }

    fn schedule_retry(&mut self, id: BroadcastId) {
        let count = self.retry_counts.entry(id.clone()).or_insert(0);
        *count += 1;
        // 2s, 4s, 8s, 16s, 30s cap
        let delay_secs = std::cmp::min(2u64.saturating_pow(*count), 30);
        info!("scheduling retry #{} for {id} in {delay_secs}s", *count);
        let retry_id = id;
        self.retry_timers.push(Box::pin(async move {
            tokio::time::sleep(Duration::from_secs(delay_secs)).await;
            retry_id
        }));
    }

    fn handle_retry(&mut self, id: BroadcastId) {
        let still_known = self
            .known_remote_broadcasts
            .get(&id.0)
            .map(|names| names.contains(&id.1))
            .unwrap_or(false);

        if !still_known {
            info!("retry: {id} — peer no longer broadcasting, skipping");
            self.retry_counts.remove(&id);
            return;
        }

        if self.active_subscribe.contains(&id) {
            debug!("retry: {id} — already active, skipping");
            return;
        }

        let attempt = self.retry_counts.get(&id).copied().unwrap_or(0);
        info!("retry: {id} — attempt #{attempt}, reconnecting");
        self.active_subscribe.insert(id.clone());
        let remote = id.0;
        let name = id.1.clone();
        self.start_connect_and_subscribe(id, remote, name);
    }

    /// Periodic reconciliation: reconnect any known broadcasts that are not actively subscribed.
    fn reconcile_subscriptions(&mut self) {
        // Collect missing subscriptions first to avoid borrow conflict.
        let active = &self.active_subscribe;
        let mut missing = Vec::new();
        for (remote, names) in &self.known_remote_broadcasts {
            for name in names {
                let id = BroadcastId(*remote, name.clone());
                if !active.contains(&id) {
                    missing.push((*remote, name.clone()));
                }
            }
        }

        for (remote, name) in missing {
            let id = BroadcastId(remote, name.clone());
            info!("reconcile: {id} — not active, reconnecting");
            self.active_subscribe.insert(id.clone());
            self.start_connect_and_subscribe(id, remote, name);
        }
    }

    async fn update_kv(&self) {
        let state = PeerState {
            broadcasts: self.active_publish.iter().cloned().collect(),
        };
        if let Err(err) = self
            .kv_writer
            .put(PEER_STATE_KEY, postcard::to_stdvec(&state).unwrap())
            .await
        {
            warn!("failed to update gossip kv: {err:#}");
        }
    }
}

mod ticket {
    use std::str::FromStr;

    use iroh::EndpointId;
    use iroh_gossip::TopicId;
    use n0_error::{Result, StdResultExt};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize, Clone, derive_more::Display)]
    #[display("{}", iroh_tickets::Ticket::serialize(self))]
    pub struct RoomTicket {
        pub bootstrap: Vec<EndpointId>,
        pub topic_id: TopicId,
    }

    impl RoomTicket {
        pub fn new(topic_id: TopicId, bootstrap: impl IntoIterator<Item = EndpointId>) -> Self {
            Self {
                bootstrap: bootstrap.into_iter().collect(),
                topic_id,
            }
        }

        pub fn generate() -> Self {
            Self {
                bootstrap: vec![],
                topic_id: TopicId::from_bytes(rand::random()),
            }
        }

        pub fn new_from_env() -> Result<Self> {
            if let Ok(value) = std::env::var("IROH_LIVE_ROOM") {
                value
                    .parse()
                    .std_context("failed to parse ticket from IROH_LIVE_ROOM environment variable")
            } else {
                let topic_id = match std::env::var("IROH_LIVE_TOPIC") {
                    Ok(topic) => TopicId::from_bytes(
                        data_encoding::HEXLOWER
                            .decode(topic.as_bytes())
                            .std_context("invalid hex")?
                            .as_slice()
                            .try_into()
                            .std_context("invalid length")?,
                    ),
                    Err(_) => {
                        let topic = TopicId::from_bytes(rand::random());
                        println!(
                            "Created new topic. Reuse with IROH_TOPIC={}",
                            data_encoding::HEXLOWER.encode(topic.as_bytes())
                        );
                        topic
                    }
                };
                Ok(Self::new(topic_id, vec![]))
            }
        }
    }

    impl FromStr for RoomTicket {
        type Err = iroh_tickets::ParseError;

        fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
            iroh_tickets::Ticket::deserialize(s)
        }
    }

    impl iroh_tickets::Ticket for RoomTicket {
        const KIND: &'static str = "room";

        fn to_bytes(&self) -> Vec<u8> {
            postcard::to_stdvec(self).unwrap()
        }

        fn from_bytes(bytes: &[u8]) -> Result<Self, iroh_tickets::ParseError> {
            let ticket = postcard::from_bytes(bytes)?;
            Ok(ticket)
        }
    }
}
