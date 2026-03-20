# iroh-live-patched/

**Generated:** 2026-03-20

## Overview

Patched iroh-live streaming libs. Workspace containing 3 crates that are pulled into `alexandria/` via `[patch]` in workspace `Cargo.toml`.

## WORKSPACE MEMBERS

| Crate | Purpose |
|-------|---------|
| `iroh-live/` | Live streaming core |
| `iroh-moq/` | MoQ protocol implementation |
| `moq-media/` | Media processing |

## USAGE IN ALEXANDRIA

The main app patches these from `iroh-live-patched/` via:

```toml
[patch]
iroh-live = { git = "https://github.com/ifftu-dev/iroh-live", rev = "..." }
# OR local path replacement
```

Features (platform-conditional):
- `tutoring-video` — desktop (ffmpeg, nokhwa)
- `tutoring-video-static` — Windows/Linux (statically linked ffmpeg)
- `tutoring-video-ios` — iOS (VideoToolbox)

## BUILD

```bash
cargo build --release  # From iroh-live-patched/
# OR
cargo build  # From alexandria/ (uses patched deps)
```
