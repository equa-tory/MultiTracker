# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MultiTracker is a single-file desktop Tkinter application (`multitrack_player.py`) for
synchronized playback of multitrack audio recordings (e.g. live show multitracks) with
per-track mute/volume, group muting, waveform display, and position "flags" (cue points).
There is no build system, package, or test suite — it's a ~970-line script run directly.

## Running

```
venv\Scripts\python.exe multitrack_player.py
```
or on Windows, double-click / run `start.bat`, which invokes the above.

Dependencies are pinned in `requirements.txt` and installed into `venv/`. Core deps:
`sounddevice`, `soundfile`, `numpy`. Optional, feature-gated at import time with `_SCIPY`/
`_LIBROSA`/`_DND` flags (each wrapped in `try/except ImportError`):
- `scipy` — polyphase resampling (`resample_poly`); falls back to `np.interp` if absent.
- `librosa` — fallback decoder for formats `soundfile` can't read (e.g. MP3/AAC).
- `tkinterdnd2` — drag-and-drop of files/folders onto the window.

There are no lint/test/build commands configured in this repo — verify changes by running
the app and exercising the UI manually.

## Architecture

Everything lives in `multitrack_player.py`, organized top-to-bottom into clearly commented
sections (`# ── section ──` banners). Read a whole section rather than grepping for a single
symbol, since related state is tightly coupled within it:

1. **Helpers** (`_peaks`, `mk_btn`, `fmt`, `parse_dnd`) — small free functions used throughout.
2. **Model** (`Group`, `Track`) — plain data holders. `Track.effective_mute` is the key piece
   of logic: individual mute wins, then `unmute_override` (track overrides its group's mute),
   then the group's own mute state. This same 4-state precedence is mirrored in the UI mute
   button (`TrackRow._toggle_mute`/`_sync`) and must stay in sync if changed.
3. **Engine** — owns all decoded audio (`Track.data`, resampled to a fixed `SR=44100`/`NCH=2`
   float32 buffer) and drives a single `sounddevice.OutputStream`. `_cb` is the realtime audio
   callback: it must stay allocation-light on the hot path (pre-sized `self._buf` scratch
   buffer, no Python-level per-sample work) since it runs on the audio thread. Playback
   position (`_frame`), seeking (`_seek`), and the 9 "flag" cue points (`flags`) all live here.
   Track loading (`add_track`) always converts to stereo and resamples to `SR` up front so the
   mix callback can just slice-and-add.
4. **UI widgets** — `WaveformCanvas` (static peak line drawn once, playhead moved via
   `coords()` with no reallocation per frame — see its docstring for the perf rationale),
   `SeekBar`, `GroupHeader`, `TrackRow`. These are plain `tk`/`ttk` widgets, no framework.
5. **App** — wires everything together: toolbar, scrollable track list (`Canvas` + inner
   `Frame`, manually managed scrollregion), transport controls, flag buttons, keyboard
   shortcuts (`_on_key`), and a `root.after(POLL_MS, ...)` poll loop (`_poll`) that pushes
   engine state into widgets each tick — this is the only place widgets are updated from
   engine state, there's no separate event/observer system.

### Track loading & caching

Files are decoded in parallel via `ThreadPoolExecutor` (`_ingest` in `App`, and the analogous
loader in `_open_proj`). Decoded audio is cached as `.npz` (raw resampled float32 data + peak
array) keyed by `md5(path:mtime:SR)`:
- Loading a **folder** or **drag-and-drop folder** caches into `<folder>/.multitrack_cache/`.
- Loading a **project** (`.mtproj`) caches into `<project_dir>/.<project_stem>_cache/`, written
  on both save and load.
Both loaders duplicate the same load/cache logic (`load_one` closures) — if you change one,
check whether the other needs the same fix.

### Project files (`.mtproj`)

JSON (see `_save`/`_open_proj`): playback position, master volume, per-track state
(path/group/muted/unmute_override/volume), per-group state (muted/collapsed), 9 flag
positions (stored in seconds, converted to frames on load), window geometry, and a relative
cache-dir hint. `version` field is present but there's no migration logic yet — bump/handle
it if the schema changes.

### Mute button state machine

The mute button (`TrackRow`) cycles through 4 visual/logical states — dim (inactive) → red
(individually muted) → orange (muted via group, can override) → green ↑ (overriding group
mute). The precedence rules are documented in the module docstring at the top of
`multitrack_player.py` and implemented in both `Track.effective_mute` and
`TrackRow._toggle_mute`/`_sync` — keep these three in agreement when touching mute logic.
