#!/usr/bin/env python3
"""
MultiTrack Player — Perfect-sync multi-track audio player
══════════════════════════════════════════════════════════

  pip install sounddevice soundfile numpy
  pip install scipy        # optional – better resampling
  pip install librosa      # optional – MP3/AAC support
  pip install tkinterdnd2  # optional – drag-and-drop

  Keyboard shortcuts
  Space / ← → / Shift+←→ / Ctrl+←→ / Home End
  1-9  jump to flag      Shift+1-9  set flag
"""

import json
from math import gcd
from pathlib import Path

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

# ── audio ──────────────────────────────────────────────────────────────────
SR        = 44100
NCH       = 2
BLOCKSIZE = 512
POLL_MS   = 40
NFLAGS    = 9
AUDIO_EXTS = {".wav",".flac",".ogg",".aiff",".aif",".mp3",".m4a",".opus",".w64"}

# ── colours ────────────────────────────────────────────────────────────────
BG      = "#111120"
BG2     = "#1a1a30"
BG3     = "#242440"
BG4     = "#30305a"
BG5     = "#44447a"
FG      = "#ddddf8"
FG2     = "#7777aa"
ACCENT  = "#6699ff"
GREEN   = "#44ee88"
RED     = "#ff4466"
YELLOW  = "#ffcc33"
PURPLE  = "#bb66ff"


# ══════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════

def mk_btn(parent, text, cmd, **kw):
    """
    Create a flat dark-theme button.
    Caller kwargs OVERRIDE defaults — uses dict.update() so there
    are NEVER duplicate keyword arguments passed to tk.Button.
    """
    cfg = dict(relief="flat", bd=0, padx=8, pady=4,
               cursor="hand2", bg=BG3, fg=FG,
               font=("monospace", 9),
               activebackground=BG4, activeforeground=FG)
    cfg.update(kw)          # ← overrides, never duplicates
    return tk.Button(parent, text=text, command=cmd, **cfg)


def hsep(parent):
    tk.Frame(parent, bg=BG4, height=1).pack(fill="x")


def fmt(sec: float) -> str:
    s = int(sec); return f"{s // 60}:{s % 60:02d}"


def trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n-1] + "…"


def parse_dnd(data: str) -> list[str]:
    paths, data = [], data.strip()
    while data:
        if data.startswith("{"):
            end = data.index("}")
            paths.append(data[1:end]); data = data[end+1:].strip()
        else:
            parts = data.split(None, 1)
            paths.append(parts[0]); data = parts[1].strip() if len(parts) > 1 else ""
    return paths


# ══════════════════════════════════════════════════════════════════════════
#  Data model
# ══════════════════════════════════════════════════════════════════════════

class Group:
    def __init__(self, name):
        self.name  = name
        self.muted = False

class Track:
    def __init__(self, path, data, group):
        self.path   = path
        self.name   = Path(path).stem
        self.data   = data          # (N,2) float32
        self.group  = group
        self.muted  = False
        self.volume = 1.0

    @property
    def effective_mute(self):
        return self.muted or self.group.muted


# ══════════════════════════════════════════════════════════════════════════
#  Engine  (one callback → perfect sync)
# ══════════════════════════════════════════════════════════════════════════

class Engine:
    def __init__(self):
        self.tracks      = []
        self.groups      = {}
        self.total_frames= 0
        self.master_vol  = 1.0
        self._frame      = 0
        self._playing    = False
        self._seek       = None
        self.flags       = [None] * NFLAGS
        self._stream     = None
        self._buf        = np.zeros((BLOCKSIZE + 256, NCH), dtype=np.float32)

    @property
    def frame(self)   : return self._frame
    @property
    def playing(self) : return self._playing
    @property
    def pos_sec(self) : return self._frame / SR
    @property
    def dur_sec(self) : return self.total_frames / SR

    # loading
    @staticmethod
    def _read(path):
        try:
            return sf.read(path, dtype="float32", always_2d=True)
        except Exception as e:
            if not _LIBROSA:
                raise RuntimeError(f"Can't read {Path(path).name}: {e}\n"
                                   "pip install librosa  for MP3/AAC") from e
            d, r = librosa.load(path, sr=None, mono=False, dtype=np.float32)
            if d.ndim == 1: d = d[np.newaxis, :]
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
        n  = int(len(d) * dst / src)
        xs = np.arange(len(d), dtype=np.float64)
        xn = np.linspace(0, len(d)-1, n)
        out = np.empty((n, d.shape[1]), dtype=np.float32)
        for c in range(d.shape[1]):
            out[:, c] = np.interp(xn, xs, d[:, c])
        return out

    def add_track(self, path, group_name="Default"):
        d, sr = self._read(path)
        d = self._stereo(d)
        d = self._resample(d, sr, SR)
        d = np.ascontiguousarray(d, dtype=np.float32)
        if group_name not in self.groups:
            self.groups[group_name] = Group(group_name)
        t = Track(str(path), d, self.groups[group_name])
        self.tracks.append(t)
        self.total_frames = max(self.total_frames, len(d))
        return t

    def remove_track(self, t):
        if t in self.tracks: self.tracks.remove(t)
        self.total_frames = max((len(x.data) for x in self.tracks), default=0)

    # callback
    def _cb(self, outdata, frames, _t, _s):
        if frames > self._buf.shape[0]:
            self._buf = np.zeros((frames+64, NCH), dtype=np.float32)
        s = self._seek
        if s is not None:
            self._seek = None; self._frame = s
        pos = self._frame
        if not self._playing or pos >= self.total_frames:
            outdata.fill(0.0); return
        buf = self._buf[:frames]; buf.fill(0.0)
        mv  = self.master_vol
        for t in self.tracks[:]:
            if t.effective_mute: continue
            end = min(pos+frames, len(t.data)); n = end-pos
            if n <= 0: continue
            buf[:n] += t.data[pos:end] * (t.volume * mv)
        np.clip(buf, -1.0, 1.0, out=buf)
        outdata[:] = buf
        self._frame += frames
        if self._frame >= self.total_frames:
            self._playing = False; self._frame = self.total_frames

    def _open(self):
        if self._stream and self._stream.active: return
        if self._stream: self._stream.close()
        self._stream = sd.OutputStream(
            samplerate=SR, channels=NCH, dtype="float32",
            blocksize=BLOCKSIZE, callback=self._cb)
        self._stream.start()

    def play(self):
        if self.tracks: self._open(); self._playing = True
    def pause(self):   self._playing = False
    def toggle(self):  (self.pause if self._playing else self.play)()
    def seek(self, f):
        f = max(0, min(int(f), self.total_frames))
        self._seek = f; self._frame = f
    def seek_sec(self, s):   self.seek(int(s * SR))
    def seek_rel(self, ds):  self.seek_sec(self.pos_sec + ds)
    def set_flag(self, i):   self.flags[i] = self._frame
    def goto_flag(self, i):
        if self.flags[i] is not None: self.seek(self.flags[i])
    def close(self):
        self._playing = False
        if self._stream: self._stream.stop(); self._stream.close(); self._stream = None


# ══════════════════════════════════════════════════════════════════════════
#  SeekBar
# ══════════════════════════════════════════════════════════════════════════

class SeekBar(tk.Canvas):
    PAD = 28
    def __init__(self, parent, engine, **kw):
        super().__init__(parent, bg=BG, height=58, highlightthickness=0,
                         cursor="hand2", **kw)
        self.engine = engine; self._drag = False
        self.bind("<Configure>",       self._draw)
        self.bind("<ButtonPress-1>",   self._press)
        self.bind("<B1-Motion>",       self._motion)
        self.bind("<ButtonRelease-1>", lambda _: setattr(self,"_drag",False))

    def _frac(self, x):
        w = self.winfo_width()
        return max(0.0, min(1.0, (x-self.PAD) / max(1, w-2*self.PAD)))
    def _tx(self, frame):
        w = self.winfo_width()
        return self.PAD + (frame/max(1,self.engine.total_frames))*(w-2*self.PAD)
    def _press(self, e):
        self._drag = True
        self.engine.seek(int(self._frac(e.x)*self.engine.total_frames))
    def _motion(self, e):
        if self._drag: self.engine.seek(int(self._frac(e.x)*self.engine.total_frames))
    def update(self): self._draw()
    def _draw(self, *_):
        self.delete("all")
        w = self.winfo_width()
        if w < 30: return
        MID, BH = 36, 6
        self.create_rectangle(self.PAD, MID-BH, w-self.PAD, MID+BH, fill=BG4, outline="")
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


# ══════════════════════════════════════════════════════════════════════════
#  GroupHeader
# ══════════════════════════════════════════════════════════════════════════

class GroupHeader(tk.Frame):
    def __init__(self, parent, group, on_refresh, **kw):
        super().__init__(parent, bg=BG3, **kw)
        self.group       = group
        self._on_refresh = on_refresh

        tk.Label(self, text=f"  ▸  {group.name}",
                 bg=BG3, fg=GREEN, font=("monospace", 10, "bold")
                 ).pack(side="left", pady=4, padx=4)

        self._mb = mk_btn(self, "MUTE GROUP", self._toggle,
                          font=("monospace", 8), padx=10, pady=2)
        self._mb.pack(side="right", padx=8, pady=4)
        self._sync()

    def _toggle(self):
        self.group.muted = not self.group.muted
        self._sync(); self._on_refresh()

    def _sync(self):
        self._mb.configure(bg=RED if self.group.muted else BG4,
                           fg=BG  if self.group.muted else FG)

    def refresh(self): self._sync()


# ══════════════════════════════════════════════════════════════════════════
#  TrackRow  — every button uses mk_btn() which never duplicates kwargs
# ══════════════════════════════════════════════════════════════════════════

class TrackRow(tk.Frame):
    def __init__(self, parent, track, engine, on_rebuild, on_refresh, **kw):
        super().__init__(parent, bg=BG2, **kw)
        self.track    = track
        self.engine   = engine
        self._rebuild = on_rebuild
        self._refresh = on_refresh
        self._build()

    def _build(self):
        # Mute button
        self._mb = mk_btn(self, "M", self._toggle_mute,
                          width=2, padx=4, pady=3)
        self._mb.pack(side="left", padx=(6,2), pady=3)

        # Track name
        tk.Label(self, text=self.track.name,
                 bg=BG2, fg=FG, font=("monospace", 9),
                 anchor="w", width=24
                 ).pack(side="left", padx=4)

        # Remove button
        mk_btn(self, "✕", self._remove,
               fg=FG2, padx=6, pady=3
               ).pack(side="right", padx=(0,6), pady=3)

        # Group badge
        self._gb = mk_btn(self, trunc(self.track.group.name, 14),
                          self._edit_group,
                          fg=GREEN, font=("monospace", 8), padx=6, pady=2)
        self._gb.pack(side="right", padx=2, pady=3)

        # Volume % label
        self._vl = tk.Label(self, text="100%", bg=BG2, fg=FG2,
                             font=("monospace", 8), width=4)
        self._vl.pack(side="right", padx=(0,2))

        # Volume slider
        self._vv = tk.DoubleVar(value=self.track.volume)
        ttk.Scale(self, from_=0.0, to=1.5, variable=self._vv,
                  orient="horizontal", length=80,
                  command=self._on_vol
                  ).pack(side="right", padx=4)

        self._sync()

    def _toggle_mute(self):
        self.track.muted = not self.track.muted
        self._sync(); self._refresh()

    def _sync(self):
        on = self.track.effective_mute
        self._mb.configure(bg=RED if on else BG4, fg=BG if on else FG2,
                           text="M")

    def _on_vol(self, v):
        v = float(v)
        self.track.volume = v
        self._vl.configure(text=f"{int(v*100)}%")

    def _edit_group(self):
        name = simpledialog.askstring(
            "Group", "Group name:", initialvalue=self.track.group.name,
            parent=self.winfo_toplevel())
        if name:
            name = name.strip() or "Default"
            if name not in self.engine.groups:
                self.engine.groups[name] = Group(name)
            self.track.group = self.engine.groups[name]
            self._gb.configure(text=trunc(name, 14))
            self.winfo_toplevel().after(1, self._rebuild)

    def _remove(self):
        self.engine.remove_track(self.track)
        self.winfo_toplevel().after(1, self._rebuild)

    def refresh(self): self._sync()


# ══════════════════════════════════════════════════════════════════════════
#  Application
# ══════════════════════════════════════════════════════════════════════════

class App:
    def __init__(self):
        self.engine = Engine()
        self.root   = TkinterDnD.Tk() if _DND else tk.Tk()
        self._rows  = []
        self._ghdrs = {}
        self._setup_style()
        self._build_ui()
        self._bind_keys()
        self._bind_dnd()
        self._poll()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_style(self):
        s = ttk.Style(self.root)
        s.theme_use("clam")
        s.configure("TScale",              background=BG2, troughcolor=BG4, sliderlength=14)
        s.configure("Vertical.TScrollbar", background=BG3, troughcolor=BG2,
                    arrowcolor=FG2, bordercolor=BG2, lightcolor=BG3, darkcolor=BG3)

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        r = self.root
        r.title("MultiTrack Player")
        r.configure(bg=BG)
        r.geometry("960x700")
        r.minsize(700, 460)

        # ── toolbar ──
        tb = tk.Frame(r, bg=BG, pady=6)
        tb.pack(fill="x", padx=10)

        mk_btn(tb, "＋ Files",   self._add_files ).pack(side="left", padx=2)
        mk_btn(tb, "📁 Folder",  self._add_folder).pack(side="left", padx=2)
        tk.Frame(tb, bg=BG5, width=1).pack(side="left", fill="y", padx=8, pady=3)
        mk_btn(tb, "💾 Save",    self._save      ).pack(side="left", padx=2)
        mk_btn(tb, "📂 Open",    self._open_proj ).pack(side="left", padx=2)
        tk.Frame(tb, bg=BG5, width=1).pack(side="left", fill="y", padx=8, pady=3)
        mk_btn(tb, "Clear All",  self._clear, fg=RED).pack(side="left", padx=2)

        hint = "Space ▶/⏸  ←/→±5s  Shift±30s  1-9 flag  Shift+1-9 set"
        if _DND: hint += "   ✦ drag files/folders"
        tk.Label(tb, text=hint, bg=BG, fg=BG5, font=("monospace",8)
                 ).pack(side="right", padx=6)

        hsep(r)

        # ── track list ──
        list_frame = tk.Frame(r, bg=BG)
        list_frame.pack(fill="both", expand=True)

        self._cv = tk.Canvas(list_frame, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self._cv.yview)
        self._cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._cv.pack(side="left", fill="both", expand=True)

        self._inner = tk.Frame(self._cv, bg=BG)
        self._cwin  = self._cv.create_window((0,0), window=self._inner, anchor="nw")

        self._inner.bind("<Configure>",
            lambda _: self._cv.configure(scrollregion=self._cv.bbox("all")))
        self._cv.bind("<Configure>",
            lambda e: self._cv.itemconfig(self._cwin, width=e.width))
        self._cv.bind_all("<MouseWheel>",
            lambda e: self._cv.yview_scroll(-1*(e.delta//120), "units"))
        self._cv.bind_all("<Button-4>", lambda _: self._cv.yview_scroll(-1,"units"))
        self._cv.bind_all("<Button-5>", lambda _: self._cv.yview_scroll(+1,"units"))

        hsep(r)

        # ── seek bar ──
        self._sb = SeekBar(r, self.engine)
        self._sb.pack(fill="x")
        hsep(r)

        # ── transport ──
        tp = tk.Frame(r, bg=BG, pady=6)
        tp.pack(fill="x", padx=10)

        self._pb = mk_btn(tp, "▶  Play", self.engine.toggle,
                          font=("monospace", 11, "bold"))
        self._pb.pack(side="left", padx=(0,8))

        for label, delta in [("⏮", None), ("◀◀ −30s",-30), ("◀ −5s",-5),
                              ("▶ +5s",+5), ("▶▶ +30s",+30)]:
            cmd = (lambda: self.engine.seek(0)) if delta is None \
                  else (lambda d=delta: self.engine.seek_rel(d))
            mk_btn(tp, label, cmd).pack(side="left", padx=2)

        self._tv = tk.StringVar(value="0:00 / 0:00")
        tk.Label(tp, textvariable=self._tv, bg=BG, fg=ACCENT,
                 font=("monospace", 15, "bold"), width=14
                 ).pack(side="left", padx=14)

        # master vol
        self._mv_lbl = tk.Label(tp, text="100%", bg=BG, fg=FG2,
                                 font=("monospace",8), width=4)
        self._mv_lbl.pack(side="right", padx=(0,2))
        self._mv = tk.DoubleVar(value=1.0)
        ttk.Scale(tp, from_=0.0, to=1.5, variable=self._mv,
                  orient="horizontal", length=90,
                  command=lambda v: setattr(self.engine,"master_vol",float(v))
                  ).pack(side="right", padx=2)
        tk.Label(tp, text="Vol:", bg=BG, fg=FG2, font=("monospace",8)
                 ).pack(side="right", padx=(0,2))

        hsep(r)

        # ── flags: two rows (SET + GO) ──
        flag_frame = tk.Frame(r, bg=BG, pady=2)
        flag_frame.pack(fill="x", padx=10, pady=(2,6))

        set_row = tk.Frame(flag_frame, bg=BG)
        set_row.pack(fill="x", pady=(0,2))
        tk.Label(set_row, text="Set flag:", bg=BG, fg=FG2,
                 font=("monospace",8), width=8, anchor="e"
                 ).pack(side="left", padx=(0,4))
        self._set_fbs = []
        for i in range(NFLAGS):
            b = tk.Button(set_row, text=str(i+1), width=2,
                          bg=BG4, fg=FG2, relief="flat", bd=0,
                          padx=4, pady=2, cursor="hand2",
                          font=("monospace",9),
                          activebackground=BG5, activeforeground=FG,
                          command=lambda idx=i: self._set_flag(idx))
            b.pack(side="left", padx=2)
            self._set_fbs.append(b)
        tk.Label(set_row, text="  click to pin current position",
                 bg=BG, fg=BG5, font=("monospace",8)).pack(side="left", padx=4)

        go_row = tk.Frame(flag_frame, bg=BG)
        go_row.pack(fill="x")
        tk.Label(go_row, text="Go to:", bg=BG, fg=FG2,
                 font=("monospace",8), width=8, anchor="e"
                 ).pack(side="left", padx=(0,4))
        self._fbs = []
        for i in range(NFLAGS):
            b = tk.Button(go_row, text=str(i+1), width=2,
                          bg=BG3, fg=YELLOW, relief="flat", bd=0,
                          padx=4, pady=2, cursor="hand2",
                          font=("monospace",9,"bold"),
                          activebackground=BG4, activeforeground=YELLOW,
                          command=lambda idx=i: self.engine.goto_flag(idx))
            b.pack(side="left", padx=2)
            self._fbs.append(b)
        tk.Label(go_row, text="  jump (dims when unset)",
                 bg=BG, fg=BG5, font=("monospace",8)).pack(side="left", padx=4)

    # ── track list ───────────────────────────────────────────────────────────
    def rebuild(self):
        for w in self._inner.winfo_children():
            w.destroy()
        self._rows.clear(); self._ghdrs.clear()

        if not self.engine.tracks:
            msg = "No tracks  —  ＋ Files  /  📁 Folder"
            if _DND: msg += "  /  drag here"
            tk.Label(self._inner, text=f"\n  {msg}\n",
                     bg=BG, fg=BG5, font=("monospace",10)
                     ).pack(anchor="w", padx=16, pady=12)
            return

        by_group = {}
        for t in self.engine.tracks:
            by_group.setdefault(t.group.name, []).append(t)

        for gname, tracks in by_group.items():
            group = self.engine.groups[gname]
            hdr   = GroupHeader(self._inner, group, self._refresh_rows)
            hdr.pack(fill="x", pady=(8,0))
            self._ghdrs[gname] = hdr

            for t in tracks:
                row = TrackRow(self._inner, t, self.engine,
                               on_rebuild=self.rebuild,
                               on_refresh=self._refresh_rows)
                row.pack(fill="x", padx=(24,4), pady=1)
                self._rows.append(row)

        tk.Frame(self._inner, bg=BG, height=12).pack()

    def _refresh_rows(self):
        for r in self._rows:            r.refresh()
        for h in self._ghdrs.values(): h.refresh()

    # ── load ────────────────────────────────────────────────────────────────
    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select audio files",
            filetypes=[("Audio","*.wav *.flac *.ogg *.aiff *.aif *.mp3 *.m4a *.opus *.w64"),
                       ("All","*.*")])
        if paths: self._ingest(list(paths))

    def _add_folder(self):
        folder = filedialog.askdirectory(title="Select folder")
        if not folder: return
        paths = sorted(str(p) for p in Path(folder).rglob("*")
                       if p.suffix.lower() in AUDIO_EXTS and p.is_file())
        if not paths:
            messagebox.showwarning("Empty","No audio files found."); return
        self._ingest(paths, base=folder)

    def _ingest(self, paths, base=""):
        errors = []
        for p in paths:
            if Path(p).suffix.lower() not in AUDIO_EXTS: continue
            try:
                grp = Path(p).parent.name if base else "Default"
                self.engine.add_track(p, grp)
            except Exception as e:
                errors.append(f"{Path(p).name}: {e}")
        self.rebuild()
        if errors:
            messagebox.showerror(f"Load errors ({len(errors)})",
                                 "\n".join(errors[:15]))

    def _clear(self):
        if messagebox.askyesno("Clear","Remove all tracks?"):
            self.engine.pause()
            self.engine.tracks.clear()
            self.engine.groups.clear()
            self.engine.total_frames = 0
            self.engine.seek(0)
            self.rebuild()

    # ── drag-and-drop ────────────────────────────────────────────────────────
    def _bind_dnd(self):
        if not _DND: return
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", self._on_drop)

    def _on_drop(self, event):
        files = []; base = ""
        for p in parse_dnd(event.data):
            path = Path(p)
            if path.is_dir():
                found = sorted(str(f) for f in path.rglob("*")
                               if f.suffix.lower() in AUDIO_EXTS and f.is_file())
                files.extend(found)
                if found: base = str(path)
            elif path.is_file() and path.suffix.lower() in AUDIO_EXTS:
                files.append(str(path))
        if files: self._ingest(files, base=base)

    # ── project ──────────────────────────────────────────────────────────────
    def _save(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".mtproj",
            filetypes=[("Project","*.mtproj"),("JSON","*.json")])
        if not p: return
        Path(p).write_text(json.dumps({
            "tracks": [{"path":t.path,"group":t.group.name,
                        "muted":t.muted,"volume":t.volume}
                       for t in self.engine.tracks],
            "flags":  self.engine.flags,
            "groups": {n:{"muted":g.muted} for n,g in self.engine.groups.items()},
        }, indent=2))
        messagebox.showinfo("Saved", p)

    def _open_proj(self):
        p = filedialog.askopenfilename(
            filetypes=[("Project","*.mtproj"),("JSON","*.json")])
        if not p: return
        try: data = json.loads(Path(p).read_text())
        except Exception as e: messagebox.showerror("Error",str(e)); return
        self.engine.pause()
        self.engine.tracks.clear(); self.engine.groups.clear()
        self.engine.total_frames = 0
        for td in data.get("tracks",[]):
            try:
                t = self.engine.add_track(td["path"], td.get("group","Default"))
                t.muted = td.get("muted",False); t.volume = td.get("volume",1.0)
            except Exception: pass
        for n,gd in data.get("groups",{}).items():
            if n in self.engine.groups:
                self.engine.groups[n].muted = gd.get("muted",False)
        fl = data.get("flags",[None]*NFLAGS)
        self.engine.flags = (list(fl)+[None]*NFLAGS)[:NFLAGS]
        self.engine.seek(0); self.rebuild()

    def _set_flag(self, idx):
        """Pin a flag at the current position and flash the Set button."""
        self.engine.set_flag(idx)
        b = self._set_fbs[idx]
        b.configure(bg=YELLOW, fg=BG)
        self.root.after(300, lambda: b.configure(bg=BG4, fg=FG2))

    # ── keyboard ─────────────────────────────────────────────────────────────
    # Shift+numrow produces DIFFERENT keysyms on US layout (exclam, at, …).
    # Map them explicitly so keyboard shortcuts actually work.
    _SHIFT_NUMS = {
        "exclam":0, "at":1, "numbersign":2, "dollar":3, "percent":4,
        "asciicircum":5, "ampersand":6, "asterisk":7, "parenleft":8,
    }

    def _bind_keys(self):
        self.root.bind_all("<KeyPress>", self._on_key)

    def _on_key(self, e):
        if isinstance(e.widget, (tk.Entry, tk.Text)):
            return

        ctrl = bool(e.state & 0x0004)
        shift = bool(e.state & 0x0001)
        key  = e.keysym

        if key == "space":
            self.engine.toggle()
        elif key == "Left":
            self.engine.seek_rel(-60 if ctrl else -30 if shift else -5)
        elif key == "Right":
            self.engine.seek_rel(+60 if ctrl else +30 if shift else +5)
        elif key == "Home":
            self.engine.seek(0)
        elif key == "End":
            self.engine.seek(self.engine.total_frames)
        elif key in "123456789":
            # Plain number → go to flag
            self.engine.goto_flag(int(key) - 1)
        elif key in self._SHIFT_NUMS:
            # Shift+number on US layout → set flag
            self._set_flag(self._SHIFT_NUMS[key])
        else:
            return  # don't consume unknown keys

        return "break"

    # ── poll ─────────────────────────────────────────────────────────────────
    def _poll(self):
        playing = self.engine.playing
        self._pb.configure(text="⏸ Pause" if playing else "▶  Play",
                           fg=YELLOW       if playing else FG)
        self._tv.set(f"{fmt(self.engine.pos_sec)} / {fmt(self.engine.dur_sec)}")
        self._mv_lbl.configure(text=f"{int(self._mv.get()*100)}%")
        self._sb.update()
        for i, (gb, sb) in enumerate(zip(self._fbs, self._set_fbs)):
            has = self.engine.flags[i] is not None
            # Go button: bright when flag is set, dim when not
            gb.configure(bg=YELLOW if has else BG3, fg=BG if has else FG2)
            # Set button: show time label when flag is set
            if has:
                t = fmt(self.engine.flags[i] / SR)
                sb.configure(text=f"{i+1}\n{t}", font=("monospace",7))
            else:
                sb.configure(text=str(i+1), font=("monospace",9))
        self.root.after(POLL_MS, self._poll)

    def _on_close(self):
        self.engine.close(); self.root.destroy()

    def run(self):
        self.rebuild(); self.root.mainloop()


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    App().run()