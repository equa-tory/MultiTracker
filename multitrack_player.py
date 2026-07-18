#!/usr/bin/env python3
"""
MultiTrack Player
  pip install sounddevice soundfile numpy
  pip install scipy        # better resampling
  pip install librosa      # MP3/AAC support
  pip install tkinterdnd2  # drag-and-drop

  Mute button states:
    RED  M  = individually muted
    DIM  M  = active (follows group)
   ORANGE M  = muted via group  → click to OVERRIDE (let this track play)
    GRN  ↑  = overriding group mute  → click to re-follow group
"""

import json
import hashlib
import os
from math import gcd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import sounddevice as sd
import soundfile as sf
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

try:
    from scipy.signal import resample_poly; _SCIPY = True
except ImportError:
    _SCIPY = False
try:
    import librosa; _LIBROSA = True
except ImportError:
    _LIBROSA = False
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD; _DND = True
except ImportError:
    _DND = False

SR         = 44100
NCH        = 2
BLOCKSIZE  = 512
POLL_MS    = 40
NFLAGS     = 9
AUDIO_EXTS = {".wav",".flac",".ogg",".aiff",".aif",".mp3",".m4a",".opus",".w64"}
WV_W, WV_H = 130, 28   # waveform canvas dimensions

BG     = "#111120"; BG2 = "#1a1a30"; BG3 = "#242440"
BG4    = "#30305a"; BG5 = "#44447a"
FG     = "#ddddf8"; FG2 = "#7777aa"
ACCENT = "#6699ff"; GREEN = "#44ee88"; RED = "#ff4466"
YELLOW = "#ffcc33"; PURPLE = "#bb66ff"; ORANGE = "#ffaa33"


# ── helpers ───────────────────────────────────────────────────────────────────

def _peaks(data: np.ndarray, width: int = WV_W) -> np.ndarray:
    mono = np.max(np.abs(data), axis=1)
    n = len(mono)
    if n == 0:
        return np.zeros(width, dtype=np.float32)
    pad = (-n) % width
    if pad:
        mono = np.pad(mono, (0, pad))
    return mono.reshape(width, -1).max(axis=1).astype(np.float32)

def mk_btn(parent, text, cmd, **kw):
    """Flat button — uses dict.update so caller kwargs NEVER duplicate."""
    cfg = dict(relief="flat", bd=0, padx=8, pady=4, cursor="hand2",
               bg=BG3, fg=FG, font=("monospace", 9),
               activebackground=BG4, activeforeground=FG)
    cfg.update(kw)
    return tk.Button(parent, text=text, command=cmd, **cfg)

def hsep(p): tk.Frame(p, bg=BG4, height=1).pack(fill="x")
def fmt(s):  s=int(s); return f"{s//60}:{s%60:02d}"
def trunc(s, n): return s if len(s) <= n else s[:n-1]+"…"

def parse_dnd(data):
    paths, data = [], data.strip()
    while data:
        if data.startswith("{"):
            e = data.index("}"); paths.append(data[1:e]); data = data[e+1:].strip()
        else:
            p = data.split(None, 1); paths.append(p[0]); data = p[1].strip() if len(p)>1 else ""
    return paths


# ── model ─────────────────────────────────────────────────────────────────────

class Group:
    def __init__(self, name):
        self.name      = name
        self.muted     = False
        self.collapsed = False   # persisted in save file

class Track:
    def __init__(self, path, data, group, peaks=None):
        self.path            = path
        self.name            = Path(path).stem
        self.data            = data
        self.group           = group
        self.muted           = False
        self.unmute_override = False
        self.volume          = 1.0
        self.peaks           = peaks if peaks is not None else _peaks(data)

    @property
    def effective_mute(self):
        if self.muted:            return True
        if self.unmute_override:  return False   # beats group mute
        return self.group.muted


# ── engine ────────────────────────────────────────────────────────────────────

class Engine:
    def __init__(self):
        self.tracks = []; self.groups = {}; self.total_frames = 0
        self.master_vol = 1.0
        self._frame = 0; self._playing = False; self._seek = None
        self.flags = [None]*NFLAGS; self._stream = None
        self._buf = np.zeros((BLOCKSIZE+256, NCH), dtype=np.float32)
        self.loop_enabled = True
        self.loop_start   = 0            # frame
        self.loop_end     = None         # frame, None = end of longest track

    @property
    def frame(self):    return self._frame
    @property
    def playing(self):  return self._playing
    @property
    def pos_sec(self):  return self._frame / SR
    @property
    def dur_sec(self):  return self.total_frames / SR

    @property
    def loop_a(self):
        return max(0, min(int(self.loop_start), self.total_frames))
    @property
    def loop_b(self):
        e = self.total_frames if self.loop_end is None else int(self.loop_end)
        return max(0, min(e, self.total_frames))

    @staticmethod
    def _read(path):
        try: return sf.read(path, dtype="float32", always_2d=True)
        except Exception as e:
            if not _LIBROSA:
                raise RuntimeError(f"Can't read {Path(path).name}: {e}\npip install librosa") from e
            d, r = librosa.load(path, sr=None, mono=False, dtype=np.float32)
            if d.ndim == 1: d = d[np.newaxis,:]
            return np.asfortranarray(d).T.astype(np.float32), r

    @staticmethod
    def _stereo(d):
        if d.shape[1] == 1: return np.repeat(d, 2, axis=1)
        return d[:, :2]

    @staticmethod
    def _resample(d, src, dst):
        if src == dst: return d
        if _SCIPY:
            g = gcd(dst, src)
            return resample_poly(d, dst//g, src//g, axis=0).astype(np.float32)
        n = int(len(d)*dst/src); xs = np.arange(len(d), dtype=np.float64)
        xn = np.linspace(0, len(d)-1, n)
        out = np.empty((n, d.shape[1]), dtype=np.float32)
        for c in range(d.shape[1]): out[:,c] = np.interp(xn, xs, d[:,c])
        return out

    def add_track(self, path, group_name="Default"):
        d, sr = self._read(path)
        d = self._stereo(d); d = self._resample(d, sr, SR)
        d = np.ascontiguousarray(d, dtype=np.float32)
        if group_name not in self.groups: self.groups[group_name] = Group(group_name)
        t = Track(str(path), d, self.groups[group_name])
        self.tracks.append(t); self.total_frames = max(self.total_frames, len(d))
        return t

    def remove_track(self, t):
        if t in self.tracks: self.tracks.remove(t)
        self.total_frames = max((len(x.data) for x in self.tracks), default=0)

    def _cb(self, outdata, frames, _t, _s):
        if frames > self._buf.shape[0]: self._buf = np.zeros((frames+64,NCH), dtype=np.float32)
        s = self._seek
        if s is not None: self._seek = None; self._frame = s
        if not self._playing or self._frame >= self.total_frames: outdata.fill(0.0); return

        buf = self._buf[:frames]; buf.fill(0.0); mv = self.master_vol
        looping = self.loop_enabled and (self.loop_b - self.loop_a) > 0
        seg_end = self.loop_b if looping else self.total_frames
        frame, written = self._frame, 0

        while written < frames:
            if looping and frame >= seg_end: frame = self.loop_a
            if frame >= seg_end: break
            n = min(frames - written, seg_end - frame)
            if n <= 0: break
            for t in self.tracks[:]:
                if t.effective_mute: continue
                m = min(frame+n, len(t.data)) - frame
                if m <= 0: continue
                buf[written:written+m] += t.data[frame:frame+m] * (t.volume * mv)
            written += n; frame += n

        np.clip(buf, -1.0, 1.0, out=buf); outdata[:] = buf
        self._frame = frame
        if not looping and self._frame >= self.total_frames:
            self._playing = False; self._frame = self.total_frames

    def _open(self):
        if self._stream and self._stream.active: return
        if self._stream: self._stream.close()
        self._stream = sd.OutputStream(samplerate=SR, channels=NCH, dtype="float32",
                                       blocksize=BLOCKSIZE, callback=self._cb); self._stream.start()

    def play(self):
        if self.tracks: self._open(); self._playing = True
    def pause(self):   self._playing = False
    def toggle(self):  (self.pause if self._playing else self.play)()
    def seek(self, f): f=max(0,min(int(f),self.total_frames)); self._seek=f; self._frame=f
    def seek_sec(self, s):  self.seek(int(s*SR))
    def seek_rel(self, ds): self.seek_sec(self.pos_sec+ds)
    def set_flag(self, i):  self.flags[i] = self._frame
    def goto_flag(self, i):
        if self.flags[i] is not None: self.seek(self.flags[i])

    MIN_LOOP_SEC = 0.25
    def set_loop_start(self, f):
        f = max(0, min(int(f), self.total_frames))
        min_gap = int(self.MIN_LOOP_SEC * SR)
        self.loop_start = min(f, self.loop_b - min_gap) if self.loop_b else f
        self.loop_start = max(0, self.loop_start)
    def set_loop_end(self, f):
        f = max(0, min(int(f), self.total_frames))
        min_gap = int(self.MIN_LOOP_SEC * SR)
        f = max(f, self.loop_a + min_gap)
        self.loop_end = min(f, self.total_frames)
    def reset_loop(self):    self.loop_start = 0; self.loop_end = None
    def toggle_loop(self):   self.loop_enabled = not self.loop_enabled
    def close(self):
        self._playing = False
        if self._stream: self._stream.stop(); self._stream.close(); self._stream = None


# ── waveform canvas ───────────────────────────────────────────────────────────

class WaveformCanvas(tk.Canvas):
    """
    Performance-optimised waveform display.
    - Waveform envelope is a SINGLE polygon item (not one line per pixel column),
      redrawn only on resize (debounced) or mute-colour change.
    - Playhead & overlay are pre-created items moved with coords() — zero alloc per frame.
    """
    RESIZE_DEBOUNCE_MS = 60

    def __init__(self, parent, track, **kw):
        super().__init__(parent, bg=BG3, height=WV_H, highlightthickness=0, **kw)
        self._track_peaks = track.peaks
        self._muted       = False
        self._last_frac   = 0.0
        self._resize_job  = None
        # Single polygon item holds the whole waveform envelope
        self._wv = self.create_polygon(0, 0, 0, 0, fill=ACCENT, outline="", tags="wv")
        # Pre-create overlay + playhead so update_pos never allocates
        self._ov = self.create_rectangle(0, 0, 0, WV_H, fill=BG5,
                                         stipple="gray25", outline="", state="hidden")
        self._ph = self.create_line(0, 0, 0, WV_H, fill=PURPLE, width=1)
        self.bind("<Configure>", self._on_configure)

    def _on_configure(self, _e=None):
        """Debounce resize — rebuild the polygon once resizing settles."""
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(self.RESIZE_DEBOUNCE_MS, self._draw_wv)

    def _resample_peaks(self, width):
        src = self._track_peaks
        if len(src) == 0 or width <= 0:
            return np.zeros(max(1, width), dtype=np.float32)
        return np.interp(np.linspace(0, 1, width),
                         np.linspace(0, 1, len(src)), src).astype(np.float32)

    def _draw_wv(self):
        """Rebuild the waveform polygon. Called on resize settle / mute change."""
        self._resize_job = None
        w = max(1, self.winfo_width())
        if w <= 1:
            return
        mid = WV_H // 2
        col = FG2 if self._muted else ACCENT
        amps = np.maximum(1, (self._resample_peaks(w) * (mid - 2) * 0.95)).astype(np.int32)
        xs  = np.arange(w, dtype=np.int32)
        top = np.column_stack((xs, mid - amps))
        bot = np.column_stack((xs[::-1], (mid + amps)[::-1]))
        pts = np.concatenate((top, bot)).ravel().tolist()
        self.coords(self._wv, *pts)
        self.itemconfig(self._wv, fill=col)
        # Keep overlay + playhead on top
        self.tag_raise(self._ov)
        self.tag_raise(self._ph)

    def update_pos(self, frac: float):
        """Cheap — moves two pre-existing items, no allocation."""
        self._last_frac = frac
        w = max(1, self.winfo_width())
        x = max(0, int(frac * w))
        if x > 1:
            self.coords(self._ov, 0, 0, x, WV_H)
            self.itemconfig(self._ov, state="normal")
        else:
            self.itemconfig(self._ov, state="hidden")
        self.coords(self._ph, x, 0, x, WV_H)

    def set_muted(self, muted: bool):
        if muted != self._muted:
            self._muted = muted
            self._draw_wv()   # recolour waveform only on mute change


# ── seek bar ──────────────────────────────────────────────────────────────────

class SeekBar(tk.Canvas):
    PAD        = 28
    HANDLE_HIT = 8      # px hit-test radius for loop A/B handles

    def __init__(self, parent, engine, **kw):
        super().__init__(parent, bg=BG, height=64, highlightthickness=0, cursor="hand2", **kw)
        self.engine = engine; self._drag = False; self._loop_drag = None
        self._ax = self._bx = 0
        self.bind("<Configure>", self._draw)
        self.bind("<ButtonPress-1>",   self._press)
        self.bind("<B1-Motion>",       self._motion)
        self.bind("<ButtonRelease-1>", self._release)

    def _frac(self, x):
        w = self.winfo_width()
        return max(0.0, min(1.0, (x-self.PAD)/max(1, w-2*self.PAD)))
    def _tx(self, frame):
        w = self.winfo_width()
        return self.PAD + (frame/max(1,self.engine.total_frames))*(w-2*self.PAD)

    def _press(self, e):
        if abs(e.x - self._ax) <= self.HANDLE_HIT:
            self._loop_drag = "a"
        elif abs(e.x - self._bx) <= self.HANDLE_HIT:
            self._loop_drag = "b"
        else:
            self._drag = True
            self.engine.seek(int(self._frac(e.x)*self.engine.total_frames))

    def _motion(self, e):
        if self._loop_drag == "a":
            self.engine.set_loop_start(self._frac(e.x)*self.engine.total_frames); self._draw()
        elif self._loop_drag == "b":
            self.engine.set_loop_end(self._frac(e.x)*self.engine.total_frames); self._draw()
        elif self._drag:
            self.engine.seek(int(self._frac(e.x)*self.engine.total_frames))

    def _release(self, _e):
        self._drag = False; self._loop_drag = None

    def update(self): self._draw()

    def _draw(self, *_):
        self.delete("all"); w = self.winfo_width()
        if w < 30: return
        MID, BH = 36, 6
        self.create_rectangle(self.PAD, MID-BH, w-self.PAD, MID+BH, fill=BG4, outline="")

        # loop region band
        self._ax = self._tx(self.engine.loop_a); self._bx = self._tx(self.engine.loop_b)
        band_col = GREEN if self.engine.loop_enabled else FG2
        self.create_rectangle(self._ax, MID-BH, self._bx, MID+BH,
                              fill=band_col, stipple="gray25", outline="")

        px = self._tx(self.engine.frame)
        if px > self.PAD:
            self.create_rectangle(self.PAD, MID-BH, px, MID+BH, fill=ACCENT, outline="")
        self.create_line(px, MID-16, px, MID+16, fill=PURPLE, width=2)
        self.create_oval(px-5, MID-5, px+5, MID+5, fill=PURPLE, outline=BG2, width=1)

        for i, f in enumerate(self.engine.flags):
            if f is None: continue
            fx = self._tx(f)
            self.create_polygon(fx-5,MID-BH-3, fx+5,MID-BH-3, fx,MID-BH+4, fill=YELLOW, outline="")
            self.create_text(fx, MID-BH-12, text=str(i+1), fill=YELLOW, font=("monospace",7,"bold"))

        # loop A/B handles — drawn below the bar so they don't collide with flags above it
        hy = MID+BH+4
        self.create_polygon(self._ax-6,hy, self._ax+6,hy, self._ax,hy+6, fill=band_col, outline="")
        self.create_text(self._ax, hy+13, text="A", fill=band_col, font=("monospace",8,"bold"))
        self.create_polygon(self._bx-6,hy, self._bx+6,hy, self._bx,hy+6, fill=band_col, outline="")
        self.create_text(self._bx, hy+13, text="B", fill=band_col, font=("monospace",8,"bold"))


# ── group header ──────────────────────────────────────────────────────────────

class GroupHeader(tk.Frame):
    def __init__(self, parent, group, on_refresh, **kw):
        super().__init__(parent, bg=BG3, **kw)
        self.group = group; self._on_refresh = on_refresh
        self.collapsed = group.collapsed   # restore from model
        self._rows = []

        # Collapse arrow
        self._arrow = tk.Label(self, text="▾", bg=BG3, fg=GREEN,
                               font=("monospace",13), cursor="hand2", padx=4)
        self._arrow.pack(side="left", pady=3)
        self._arrow.bind("<Button-1>", lambda _: self._toggle_collapse())

        # Group name (also clickable to collapse)
        lbl = tk.Label(self, text=group.name, bg=BG3, fg=GREEN,
                       font=("monospace",10,"bold"), cursor="hand2")
        lbl.pack(side="left", pady=3)
        lbl.bind("<Button-1>", lambda _: self._toggle_collapse())

        # Track count badge
        self._count = tk.Label(self, text="", bg=BG3, fg=FG2, font=("monospace",8))
        self._count.pack(side="left", padx=6, pady=3)

        # Mute group
        self._mb = mk_btn(self, "MUTE GROUP", self._toggle_mute,
                          font=("monospace",8), padx=10, pady=2)
        self._mb.pack(side="right", padx=8, pady=3)
        self._sync_mute()

    def set_rows(self, rows):
        self._rows = rows
        self._count.configure(text=f"({len(rows)} tracks)")

    def _toggle_collapse(self):
        self.collapsed = not self.collapsed
        self.group.collapsed = self.collapsed   # persist in model
        self._arrow.configure(text="▸" if self.collapsed else "▾")
        for row in self._rows:
            if self.collapsed: row.pack_forget()
            else:              row.pack(fill="x", padx=(24,4), pady=1)

    def _toggle_mute(self):
        self.group.muted = not self.group.muted
        self._sync_mute(); self._on_refresh()

    def _sync_mute(self):
        self._mb.configure(bg=RED if self.group.muted else BG4,
                           fg=BG  if self.group.muted else FG)

    def refresh(self): self._sync_mute()


# ── track row ─────────────────────────────────────────────────────────────────

class TrackRow(tk.Frame):
    """
    Mute button cycles based on context:
      Normal state:         M (dim)   → click → RED M  (individually muted)
      Individually muted:   M (red)   → click → dim M  (unmuted)
      Group muted, track following:
                          M (orange) → click → GRN ↑  (override: track plays)
      Overriding group:   ↑ (green)  → click → orange M (back to group mute)
    """
    def __init__(self, parent, track, engine, on_rebuild, on_refresh, show_wv=True, **kw):
        super().__init__(parent, bg=BG2, **kw)
        self.track = track; self.engine = engine
        self._rebuild = on_rebuild; self._refresh = on_refresh
        self._wv = None
        self._build(show_wv)

    def _build(self, show_wv):
        # ── left-anchored fixed items ──
        self._mb = mk_btn(self, "M", self._toggle_mute, width=2, padx=4, pady=3)
        self._mb.pack(side="left", padx=(6,3), pady=3)

        tk.Label(self, text=self.track.name, bg=BG2, fg=FG,
                 font=("monospace",9), anchor="w", width=18
                 ).pack(side="left", padx=4)

        # ── right-anchored fixed items (packed before waveform so pack
        #    allocates their space first, leaving the rest for the wave) ──
        mk_btn(self, "✕", self._remove, fg=FG2, padx=6, pady=3
               ).pack(side="right", padx=(0,6), pady=3)
        self._gb = mk_btn(self, trunc(self.track.group.name,12), self._edit_group,
                          fg=GREEN, font=("monospace",8), padx=6, pady=2)
        self._gb.pack(side="right", padx=2, pady=3)
        self._vl = tk.Label(self, text="100%", bg=BG2, fg=FG2, font=("monospace",8), width=4)
        self._vl.pack(side="right", padx=(0,2))
        self._vv = tk.DoubleVar(value=self.track.volume)
        ttk.Scale(self, from_=0.0, to=1.5, variable=self._vv,
                  orient="horizontal", length=80, command=self._on_vol
                  ).pack(side="right", padx=4)

        # ── waveform: last — fills ALL remaining horizontal space ──
        # Created lazily (only when waveforms are toggled on) since it's the laggy part.
        if show_wv:
            self._create_wv()

        self._sync()

    def _create_wv(self):
        if self._wv is None:
            self._wv = WaveformCanvas(self, self.track)
        self._wv.pack(side="left", fill="x", expand=True, padx=4, pady=2)

    def _toggle_mute(self):
        t = self.track
        if t.muted:
            # individually muted → unmute normally
            t.muted = False; t.unmute_override = False
        elif t.group.muted and t.unmute_override:
            # was overriding group mute → stop overriding (follow group again)
            t.unmute_override = False
        elif t.group.muted:
            # group muted, track following → override so THIS track plays
            t.unmute_override = True
        else:
            # active normally → mute individually
            t.muted = True
        self._sync(); self._refresh()

    def _sync(self):
        t = self.track
        if t.muted:
            self._mb.configure(bg=RED,    fg=BG,  text="M"); wv_muted = True
        elif t.unmute_override:
            self._mb.configure(bg=GREEN,  fg=BG,  text="↑"); wv_muted = False
        elif t.group.muted:
            self._mb.configure(bg=ORANGE, fg=BG,  text="M"); wv_muted = True
        else:
            self._mb.configure(bg=BG4,   fg=FG2, text="M"); wv_muted = False
        if self._wv is not None: self._wv.set_muted(wv_muted)

    def _on_vol(self, v):
        v = float(v); self.track.volume = v; self._vl.configure(text=f"{int(v*100)}%")

    def _edit_group(self):
        name = simpledialog.askstring("Group","Group name:",
                                      initialvalue=self.track.group.name,
                                      parent=self.winfo_toplevel())
        if name:
            name = name.strip() or "Default"
            if name not in self.engine.groups: self.engine.groups[name] = Group(name)
            self.track.group = self.engine.groups[name]
            self._gb.configure(text=trunc(name,12))
            self.winfo_toplevel().after(1, self._rebuild)

    def _remove(self):
        self.engine.remove_track(self.track)
        self.winfo_toplevel().after(1, self._rebuild)

    def refresh(self):   self._sync()
    def update_pos(self, frac):
        if self._wv is not None: self._wv.update_pos(frac)

    def set_waveform_visible(self, visible: bool):
        if visible:
            # Created on first use (lazy) — re-packed after the right-side items
            # so it still expands correctly.
            self._create_wv()
            self._wv.after_idle(self._wv._draw_wv)   # redraw at current width
        elif self._wv is not None:
            self._wv.pack_forget()


# ── app ───────────────────────────────────────────────────────────────────────

class App:
    def __init__(self):
        self.engine = Engine()
        self.root   = TkinterDnD.Tk() if _DND else tk.Tk()
        self._rows  = []; self._ghdrs = {}
        self._setup_style(); self._build_ui()
        self._bind_keys(); self._bind_dnd(); self._poll()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_style(self):
        s = ttk.Style(self.root); s.theme_use("clam")
        s.configure("TScale", background=BG2, troughcolor=BG4, sliderlength=14)
        s.configure("Vertical.TScrollbar", background=BG3, troughcolor=BG2,
                    arrowcolor=FG2, bordercolor=BG2, lightcolor=BG3, darkcolor=BG3)

    def _build_ui(self):
        r = self.root; r.title("MultiTrack Player")
        r.configure(bg=BG); r.geometry("1060x760"); r.minsize(740,500)

        # ── toolbar ──
        tb = tk.Frame(r, bg=BG, pady=6); tb.pack(fill="x", padx=10)
        mk_btn(tb,"＋ Files",   self._add_files ).pack(side="left",padx=2)
        mk_btn(tb,"📁 Folder",  self._add_folder).pack(side="left",padx=2)
        tk.Frame(tb,bg=BG5,width=1).pack(side="left",fill="y",padx=8,pady=3)
        mk_btn(tb,"💾 Save",    self._save      ).pack(side="left",padx=2)
        mk_btn(tb,"📂 Open",    self._open_proj ).pack(side="left",padx=2)
        tk.Frame(tb,bg=BG5,width=1).pack(side="left",fill="y",padx=8,pady=3)
        mk_btn(tb,"Mute All",   self._mute_all,   fg=RED   ).pack(side="left",padx=2)
        mk_btn(tb,"Unmute All", self._unmute_all, fg=GREEN ).pack(side="left",padx=2)
        tk.Frame(tb,bg=BG5,width=1).pack(side="left",fill="y",padx=8,pady=3)
        self._show_wv = False   # off by default — waveform rendering is the laggy part
        self._wv_btn = mk_btn(tb, "📊 Waves", self._toggle_waveforms, fg=FG2)
        self._wv_btn.pack(side="left", padx=2)
        self._loop_btn = mk_btn(tb, "🔁 Loop", self._toggle_loop, fg=GREEN)
        self._loop_btn.pack(side="left", padx=2)
        tk.Frame(tb,bg=BG5,width=1).pack(side="left",fill="y",padx=8,pady=3)
        mk_btn(tb,"Clear All",  self._clear,      fg=FG2   ).pack(side="left",padx=2)
        hint = "Space  ←/→±5s  Shift±30s  1-9 flag  Shift+1-9 set"
        if _DND: hint += "  ✦ drag"
        tk.Label(tb,text=hint,bg=BG,fg=BG5,font=("monospace",8)).pack(side="right",padx=6)
        hsep(r)

        # ── track list ──
        cont = tk.Frame(r,bg=BG); cont.pack(fill="both",expand=True)
        self._cv = tk.Canvas(cont,bg=BG,highlightthickness=0)
        vsb = ttk.Scrollbar(cont,orient="vertical",command=self._cv.yview)
        self._cv.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right",fill="y"); self._cv.pack(side="left",fill="both",expand=True)
        self._inner = tk.Frame(self._cv,bg=BG)
        self._cwin  = self._cv.create_window((0,0),window=self._inner,anchor="nw")
        self._inner.bind("<Configure>",
            lambda _: self._cv.configure(scrollregion=self._cv.bbox("all")))
        self._cv.bind("<Configure>",
            lambda e: self._cv.itemconfig(self._cwin,width=e.width))
        self._cv.bind_all("<MouseWheel>",
            lambda e: self._cv.yview_scroll(-1*(e.delta//120),"units"))
        self._cv.bind_all("<Button-4>",lambda _: self._cv.yview_scroll(-1,"units"))
        self._cv.bind_all("<Button-5>",lambda _: self._cv.yview_scroll(+1,"units"))
        hsep(r)

        # ── seek bar ──
        self._seekbar = SeekBar(r,self.engine); self._seekbar.pack(fill="x"); hsep(r)

        # ── transport ──
        tp = tk.Frame(r,bg=BG,pady=6); tp.pack(fill="x",padx=10)
        self._pb = mk_btn(tp,"▶  Play",self.engine.toggle,font=("monospace",11,"bold"))
        self._pb.pack(side="left",padx=(0,8))
        for lbl,delta in [("⏮",None),("◀◀ −30s",-30),("◀ −5s",-5),("▶ +5s",+5),("▶▶ +30s",+30)]:
            cmd = (lambda: self.engine.seek(0)) if delta is None \
                  else (lambda d=delta: self.engine.seek_rel(d))
            mk_btn(tp,lbl,cmd).pack(side="left",padx=2)
        self._tv = tk.StringVar(value="0:00 / 0:00")
        tk.Label(tp,textvariable=self._tv,bg=BG,fg=ACCENT,
                 font=("monospace",15,"bold"),width=14).pack(side="left",padx=14)
        self._mv_lbl = tk.Label(tp,text="100%",bg=BG,fg=FG2,font=("monospace",8),width=4)
        self._mv_lbl.pack(side="right",padx=(0,2))
        self._mv = tk.DoubleVar(value=1.0)
        ttk.Scale(tp,from_=0.0,to=1.5,variable=self._mv,orient="horizontal",length=90,
                  command=lambda v: setattr(self.engine,"master_vol",float(v))
                  ).pack(side="right",padx=2)
        tk.Label(tp,text="Vol:",bg=BG,fg=FG2,font=("monospace",8)).pack(side="right",padx=(0,2))
        hsep(r)

        # ── flags ──
        ff = tk.Frame(r,bg=BG,pady=2); ff.pack(fill="x",padx=10,pady=(2,6))

        sr_ = tk.Frame(ff,bg=BG); sr_.pack(fill="x",pady=(0,2))
        tk.Label(sr_,text="Set flag:",bg=BG,fg=FG2,font=("monospace",8),
                 width=8,anchor="e").pack(side="left",padx=(0,4))
        self._set_fbs = []
        for i in range(NFLAGS):
            b = tk.Button(sr_,text=str(i+1),width=2,bg=BG4,fg=FG2,relief="flat",bd=0,
                          padx=4,pady=2,cursor="hand2",font=("monospace",9),
                          activebackground=BG5,activeforeground=FG,
                          command=lambda idx=i: self._set_flag(idx))
            b.pack(side="left",padx=2); self._set_fbs.append(b)
        tk.Label(sr_,text="  click to pin current position",
                 bg=BG,fg=BG5,font=("monospace",8)).pack(side="left",padx=4)

        gr_ = tk.Frame(ff,bg=BG); gr_.pack(fill="x")
        tk.Label(gr_,text="Go to:",bg=BG,fg=FG2,font=("monospace",8),
                 width=8,anchor="e").pack(side="left",padx=(0,4))
        self._fbs = []
        for i in range(NFLAGS):
            b = tk.Button(gr_,text=str(i+1),width=2,bg=BG3,fg=YELLOW,relief="flat",bd=0,
                          padx=4,pady=2,cursor="hand2",font=("monospace",9,"bold"),
                          activebackground=BG4,activeforeground=YELLOW,
                          command=lambda idx=i: self.engine.goto_flag(idx))
            b.pack(side="left",padx=2); self._fbs.append(b)
        tk.Label(gr_,text="  jump  (dims when unset)",
                 bg=BG,fg=BG5,font=("monospace",8)).pack(side="left",padx=4)

        lp_ = tk.Frame(ff,bg=BG); lp_.pack(fill="x",pady=(2,0))
        tk.Label(lp_,text="Loop:",bg=BG,fg=FG2,font=("monospace",8),
                 width=8,anchor="e").pack(side="left",padx=(0,4))
        mk_btn(lp_,"Set Loop A",self._set_loop_a,fg=GREEN,font=("monospace",8),
               padx=6,pady=2).pack(side="left",padx=2)
        mk_btn(lp_,"Set Loop B",self._set_loop_b,fg=GREEN,font=("monospace",8),
               padx=6,pady=2).pack(side="left",padx=2)
        mk_btn(lp_,"Reset 0–100%",self._reset_loop,fg=FG2,font=("monospace",8),
               padx=6,pady=2).pack(side="left",padx=2)
        tk.Label(lp_,text="  pin current position as loop start/end",
                 bg=BG,fg=BG5,font=("monospace",8)).pack(side="left",padx=4)

    # ── track list ────────────────────────────────────────────────────────────
    def rebuild(self):
        for w in self._inner.winfo_children(): w.destroy()
        self._rows.clear(); self._ghdrs.clear()
        if not self.engine.tracks:
            msg = "No tracks  —  ＋ Files  /  📁 Folder"
            if _DND: msg += "  /  drag here"
            tk.Label(self._inner,text=f"\n  {msg}\n",
                     bg=BG,fg=BG5,font=("monospace",10)).pack(anchor="w",padx=16,pady=12)
            return
        by_group = {}
        for t in self.engine.tracks: by_group.setdefault(t.group.name,[]).append(t)
        for gname, tracks in by_group.items():
            group = self.engine.groups[gname]
            hdr = GroupHeader(self._inner,group,self._refresh_rows)
            hdr.pack(fill="x",pady=(8,0)); self._ghdrs[gname] = hdr
            trows = []
            for t in tracks:
                row = TrackRow(self._inner,t,self.engine,
                               on_rebuild=self.rebuild,on_refresh=self._refresh_rows,
                               show_wv=self._show_wv)
                row.pack(fill="x",padx=(24,4),pady=1)
                self._rows.append(row); trows.append(row)
            hdr.set_rows(trows)
        tk.Frame(self._inner,bg=BG,height=12).pack()

    def _refresh_rows(self):
        for r in self._rows:            r.refresh()
        for h in self._ghdrs.values(): h.refresh()

    # ── global mute ───────────────────────────────────────────────────────────
    def _mute_all(self):
        for t in self.engine.tracks: t.muted=True;  t.unmute_override=False
        for g in self.engine.groups.values(): g.muted=False
        self._refresh_rows()

    def _unmute_all(self):
        for t in self.engine.tracks: t.muted=False; t.unmute_override=False
        for g in self.engine.groups.values(): g.muted=False
        self._refresh_rows()

    def _toggle_waveforms(self):
        self._show_wv = not self._show_wv
        self._wv_btn.configure(fg=GREEN if self._show_wv else FG2)
        for row in self._rows:
            row.set_waveform_visible(self._show_wv)

    # ── loop ──────────────────────────────────────────────────────────────────
    def _toggle_loop(self):
        self.engine.toggle_loop()
        self._loop_btn.configure(fg=GREEN if self.engine.loop_enabled else FG2)
        self._seekbar.update()

    def _set_loop_a(self):
        self.engine.set_loop_start(self.engine.frame); self._seekbar.update()

    def _set_loop_b(self):
        self.engine.set_loop_end(self.engine.frame); self._seekbar.update()

    def _reset_loop(self):
        self.engine.reset_loop(); self._seekbar.update()

    # ── file loading ──────────────────────────────────────────────────────────
    def _add_files(self):
        paths = filedialog.askopenfilenames(title="Select audio files",
            filetypes=[("Audio","*.wav *.flac *.ogg *.aiff *.aif *.mp3 *.m4a *.opus *.w64"),
                       ("All","*.*")])
        if paths: self._ingest(list(paths))

    def _add_folder(self):
        folder = filedialog.askdirectory(title="Select folder")
        if not folder: return
        paths = sorted(str(p) for p in Path(folder).rglob("*")
                       if p.suffix.lower() in AUDIO_EXTS and p.is_file())
        if not paths: messagebox.showwarning("Empty","No audio files found."); return
        cache_dir = Path(folder) / ".multitrack_cache"
        self._ingest(paths, base=folder, cache_dir=str(cache_dir))

    def _ingest(self, paths, base="", cache_dir=None):
        """Load audio files in parallel threads, with optional .npz cache."""
        valid = [p for p in paths if Path(p).suffix.lower() in AUDIO_EXTS]
        if not valid: return

        results = [None] * len(valid)
        errors  = []

        def load_one(idx, path, grp):
            # 1. try cache
            if cache_dir:
                try:
                    key  = hashlib.md5(
                        f"{path}:{os.path.getmtime(path)}:{SR}".encode()
                    ).hexdigest()
                    cp = Path(cache_dir) / f"{key}.npz"
                    if cp.exists():
                        npz = np.load(cp)
                        return idx, npz["data"], npz["peaks"], grp, None, True
                except Exception:
                    pass
            # 2. full decode
            d, sr = Engine._read(path)
            d = Engine._stereo(d)
            d = Engine._resample(d, sr, SR)
            d = np.ascontiguousarray(d, dtype=np.float32)
            pk = _peaks(d)
            return idx, d, pk, grp, None, False

        n = len(valid)
        workers = min(8, n)
        done_count = [0]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {}
            for i, p in enumerate(valid):
                grp = Path(p).parent.name if base else "Default"
                futs[pool.submit(load_one, i, p, grp)] = p
            for fut in as_completed(futs):
                path = futs[fut]
                done_count[0] += 1
                self.root.title(f"Loading {done_count[0]}/{n}…")
                try:
                    idx, d, pk, grp, _, from_cache = fut.result()
                    results[idx] = (path, d, pk, grp)
                    # save to cache if it wasn't loaded from there
                    if cache_dir and not from_cache:
                        try:
                            key = hashlib.md5(
                                f"{path}:{os.path.getmtime(path)}:{SR}".encode()
                            ).hexdigest()
                            Path(cache_dir).mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(
                                Path(cache_dir)/f"{key}.npz", data=d, peaks=pk)
                        except Exception:
                            pass
                except Exception as e:
                    errors.append(f"{Path(path).name}: {e}")

        self.root.title("MultiTrack Player")

        # add tracks to engine in original order
        for item in results:
            if item is None: continue
            path, d, pk, grp = item
            if grp not in self.engine.groups:
                self.engine.groups[grp] = Group(grp)
            t = Track(path, d, self.engine.groups[grp], peaks=pk)
            self.engine.tracks.append(t)
            self.engine.total_frames = max(self.engine.total_frames, len(d))

        self.rebuild()
        if errors:
            messagebox.showerror(f"Errors ({len(errors)})", "\n".join(errors[:15]))

    def _clear(self):
        if messagebox.askyesno("Clear","Remove all tracks?"):
            self.engine.pause(); self.engine.tracks.clear()
            self.engine.groups.clear(); self.engine.total_frames=0
            self.engine.seek(0); self.rebuild()

    # ── drag-and-drop ─────────────────────────────────────────────────────────
    def _bind_dnd(self):
        if not _DND: return
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>",self._on_drop)

    def _on_drop(self, event):
        files=[]; base=""
        for p in parse_dnd(event.data):
            path=Path(p)
            if path.is_dir():
                found=sorted(str(f) for f in path.rglob("*")
                             if f.suffix.lower() in AUDIO_EXTS and f.is_file())
                files.extend(found)
                if found: base=str(path)
            elif path.is_file() and path.suffix.lower() in AUDIO_EXTS:
                files.append(str(path))
        if files: self._ingest(files,base=base)

    # ── project ───────────────────────────────────────────────────────────────
    def _save(self):
        p = filedialog.asksaveasfilename(defaultextension=".mtproj",
            filetypes=[("Project","*.mtproj"),("JSON","*.json")])
        if not p: return
        proj_path = Path(p)
        cache_dir = proj_path.parent / f".{proj_path.stem}_cache"

        data = {
            "version":       3,
            # playback state
            "position":      self.engine.pos_sec,
            "master_volume": float(self._mv.get()),
            # loop state
            "loop_enabled": self.engine.loop_enabled,
            "loop_start":   self.engine.loop_a / SR,
            "loop_end":     (self.engine.loop_end / SR) if self.engine.loop_end is not None else None,
            # UI state
            "show_waveforms": self._show_wv,
            "window_geometry": self.root.geometry(),
            # tracks
            "tracks": [
                {"path":            t.path,
                 "group":           t.group.name,
                 "muted":           t.muted,
                 "unmute_override": t.unmute_override,
                 "volume":          t.volume}
                for t in self.engine.tracks
            ],
            # groups (mute + collapse state)
            "groups": {
                n: {"muted": g.muted, "collapsed": g.collapsed}
                for n, g in self.engine.groups.items()
            },
            # flags (9 slots, each null or seconds float)
            "flags": [
                (f / SR if f is not None else None)
                for f in self.engine.flags
            ],
            # cache dir hint (relative)
            "cache_dir": str(cache_dir.relative_to(proj_path.parent)),
        }
        proj_path.write_text(json.dumps(data, indent=2))

        # write audio cache alongside the project for fast future loads
        cache_dir.mkdir(exist_ok=True)
        for t in self.engine.tracks:
            try:
                key = hashlib.md5(
                    f"{t.path}:{os.path.getmtime(t.path)}:{SR}".encode()
                ).hexdigest()
                cp = cache_dir / f"{key}.npz"
                if not cp.exists():
                    np.savez_compressed(cp, data=t.data, peaks=t.peaks)
            except Exception:
                pass

        messagebox.showinfo("Saved", str(proj_path))

    def _open_proj(self):
        p = filedialog.askopenfilename(
            filetypes=[("Project","*.mtproj"),("JSON","*.json")])
        if not p: return
        try:
            raw = json.loads(Path(p).read_text())
        except Exception as e:
            messagebox.showerror("Error", str(e)); return

        proj_path = Path(p)
        # Resolve cache dir
        cache_hint = raw.get("cache_dir", "")
        cache_dir  = (proj_path.parent / cache_hint) if cache_hint else None
        if cache_dir and not cache_dir.is_dir():
            cache_dir = None

        self.engine.pause()
        self.engine.tracks.clear(); self.engine.groups.clear()
        self.engine.total_frames = 0

        # Restore group metadata BEFORE loading tracks (so groups exist)
        for n, gd in raw.get("groups", {}).items():
            g = Group(n)
            g.muted     = gd.get("muted",     False)
            g.collapsed = gd.get("collapsed",  False)
            self.engine.groups[n] = g

        # Load tracks in parallel (with cache)
        track_metas = raw.get("tracks", [])
        paths  = [td["path"] for td in track_metas]
        groups = [td.get("group", "Default") for td in track_metas]

        results  = [None] * len(paths)
        errors   = []
        n_tracks = len(paths)

        def load_one(idx, path, grp):
            if cache_dir:
                try:
                    key = hashlib.md5(
                        f"{path}:{os.path.getmtime(path)}:{SR}".encode()
                    ).hexdigest()
                    cp = cache_dir / f"{key}.npz"
                    if cp.exists():
                        npz = np.load(cp)
                        return idx, npz["data"], npz["peaks"], grp, None
                except Exception:
                    pass
            d, sr = Engine._read(path)
            d = Engine._stereo(d); d = Engine._resample(d, sr, SR)
            d = np.ascontiguousarray(d, dtype=np.float32)
            return idx, d, _peaks(d), grp, None

        done_count = [0]
        with ThreadPoolExecutor(max_workers=min(8, max(1, n_tracks))) as pool:
            futs = {pool.submit(load_one, i, paths[i], groups[i]): i
                    for i in range(n_tracks)}
            for fut in as_completed(futs):
                done_count[0] += 1
                self.root.title(f"Loading {done_count[0]}/{n_tracks}…")
                try:
                    idx, d, pk, grp, _ = fut.result()
                    results[idx] = (paths[idx], d, pk, grp)
                except Exception as e:
                    errors.append(f"{Path(paths[futs[fut]]).name}: {e}")

        self.root.title("MultiTrack Player")

        for item, td in zip(results, track_metas):
            if item is None: continue
            path, d, pk, grp = item
            if grp not in self.engine.groups:
                self.engine.groups[grp] = Group(grp)
            t = Track(path, d, self.engine.groups[grp], peaks=pk)
            t.muted           = td.get("muted",           False)
            t.unmute_override = td.get("unmute_override", False)
            t.volume          = td.get("volume",          1.0)
            self.engine.tracks.append(t)
            self.engine.total_frames = max(self.engine.total_frames, len(d))

        # Restore flags (stored as seconds → convert to frames)
        raw_flags = raw.get("flags", [None]*NFLAGS)
        self.engine.flags = [
            int(f * SR) if f is not None else None
            for f in (raw_flags + [None]*NFLAGS)[:NFLAGS]
        ]

        # Restore master volume
        mv = raw.get("master_volume", 1.0)
        self._mv.set(mv); self.engine.master_vol = mv

        # Restore loop state
        self.engine.loop_enabled = raw.get("loop_enabled", True)
        self.engine.loop_start   = int(raw.get("loop_start", 0.0) * SR)
        loop_end_sec = raw.get("loop_end", None)
        self.engine.loop_end = int(loop_end_sec * SR) if loop_end_sec is not None else None
        self._loop_btn.configure(fg=GREEN if self.engine.loop_enabled else FG2)

        # Restore waveform toggle
        show_wv = raw.get("show_waveforms", False)
        self._show_wv = show_wv
        self._wv_btn.configure(fg=GREEN if show_wv else FG2)

        # Restore window geometry
        geom = raw.get("window_geometry", "")
        if geom:
            try: self.root.geometry(geom)
            except Exception: pass

        self.rebuild()
        self._seekbar.update()

        # Seek to saved position (after rebuild so stream is ready)
        pos = raw.get("position", 0.0)
        if pos: self.engine.seek_sec(pos)

        if errors:
            messagebox.showerror(f"Errors ({len(errors)})", "\n".join(errors[:15]))

    # ── flags ─────────────────────────────────────────────────────────────────
    def _set_flag(self, idx):
        self.engine.set_flag(idx)
        b = self._set_fbs[idx]; b.configure(bg=YELLOW,fg=BG)
        self.root.after(300, lambda: b.configure(bg=BG4,fg=FG2))

    # ── keyboard ──────────────────────────────────────────────────────────────
    _SHIFT_NUMS = {"exclam":0,"at":1,"numbersign":2,"dollar":3,"percent":4,
                   "asciicircum":5,"ampersand":6,"asterisk":7,"parenleft":8}

    def _bind_keys(self):
        self.root.bind_all("<KeyPress>", self._on_key)

    def _on_key(self, e):
        if isinstance(e.widget,(tk.Entry,tk.Text)): return
        ctrl=bool(e.state&0x0004); shift=bool(e.state&0x0001); key=e.keysym
        if   key=="space":  self.engine.toggle()
        elif key=="Left":   self.engine.seek_rel(-60 if ctrl else -30 if shift else -5)
        elif key=="Right":  self.engine.seek_rel(+60 if ctrl else +30 if shift else +5)
        elif key=="Home":   self.engine.seek(0)
        elif key=="End":    self.engine.seek(self.engine.total_frames)
        elif key in "123456789":      self.engine.goto_flag(int(key)-1)
        elif key in self._SHIFT_NUMS: self._set_flag(self._SHIFT_NUMS[key])
        else: return
        return "break"

    # ── poll ──────────────────────────────────────────────────────────────────
    def _poll(self):
        playing = self.engine.playing
        self._pb.configure(text="⏸ Pause" if playing else "▶  Play",
                           fg=YELLOW if playing else FG)
        self._tv.set(f"{fmt(self.engine.pos_sec)} / {fmt(self.engine.dur_sec)}")
        self._mv_lbl.configure(text=f"{int(self._mv.get()*100)}%")
        self._seekbar.update()
        frac = self.engine.frame / max(1, self.engine.total_frames)
        for row in self._rows: row.update_pos(frac)
        for i,(gb,sb) in enumerate(zip(self._fbs,self._set_fbs)):
            has = self.engine.flags[i] is not None
            gb.configure(bg=YELLOW if has else BG3, fg=BG if has else FG2)
            if has:
                sb.configure(text=f"{i+1}\n{fmt(self.engine.flags[i]/SR)}",font=("monospace",7))
            else:
                sb.configure(text=str(i+1),font=("monospace",9))
        self.root.after(POLL_MS, self._poll)

    def _on_close(self):
        self.engine.close(); self.root.destroy()

    def run(self):
        self.rebuild(); self.root.mainloop()


if __name__ == "__main__":
    App().run()