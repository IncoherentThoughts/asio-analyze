"""Browser-based GUI for asio-analyze.

Launches a tiny stdlib HTTP server on localhost and opens the page in the
default browser. No new dependencies. Same commands as the CLI:
default / background / ltv / fe55 / full.
"""

import http.server
import io
import json
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from datetime import date, datetime, timedelta
from urllib.parse import urlparse, parse_qs

from . import __version__
from . import commands as _commands
from . import get_data as _get_data
from . import segments as _segments

import numpy as np


BINARY_FILENAME_RE = re.compile(r"^(\d{8})\.0x[0-9A-Fa-f]+$")


# Maximum points (per channel) returned to the browser. Each "point" carries
# (min, max), so we send 2*MAX_POINTS floats — enough to preserve spikes when
# the source has many more samples than pixels, light enough on the wire.
MAX_PLOT_POINTS = 3000


def _downsample_minmax(values, max_points):
    """Min/max-decimate `values` to at most `max_points` bins.

    Returns (mins, maxs, t_starts, dt_bin) where each pair (mins[i], maxs[i])
    spans the bin starting at t_starts[i] (in samples). If the source is
    already small, returns the raw signal (mins == maxs).
    """
    n = len(values)
    if n == 0:
        return [], [], 0.0
    if n <= max_points:
        v = [float(x) for x in values]
        return v, v, 1.0
    # bin size in samples; ensure at least 2 per bin
    bin_size = int(np.ceil(n / max_points))
    n_bins = int(np.ceil(n / bin_size))
    pad = n_bins * bin_size - n
    arr = np.asarray(values, dtype=float)
    if pad:
        arr = np.concatenate([arr, np.full(pad, arr[-1])])
    reshaped = arr.reshape(n_bins, bin_size)
    mins = reshaped.min(axis=1).tolist()
    maxs = reshaped.max(axis=1).tolist()
    return mins, maxs, float(bin_size)


def _classify_upload(paths):
    """Split a user-selected list of paths/dirs into (binaries, rpts, ignored).

    Directories are expanded one level. Binaries match the canonical
    ``YYYYMMDD.0xHEX`` filename pattern; .rpt files are recognized by
    extension; anything else is ignored.
    """
    binaries = []
    rpts = []
    ignored = []
    expanded = []
    for p in paths:
        if not p:
            continue
        p = os.path.abspath(os.path.expanduser(p))
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                expanded.append(os.path.join(p, name))
        else:
            expanded.append(p)
    for p in expanded:
        if not os.path.isfile(p):
            continue
        name = os.path.basename(p)
        if BINARY_FILENAME_RE.match(name):
            binaries.append(p)
        elif name.lower().endswith(".rpt"):
            rpts.append(p)
        else:
            ignored.append(p)
    # de-dupe while preserving order
    seen = set()
    binaries = [x for x in binaries if not (x in seen or seen.add(x))]
    seen = set()
    rpts = [x for x in rpts if not (x in seen or seen.add(x))]
    return binaries, rpts, ignored


def _check_consecutive_dates(binaries):
    """Verify the binaries' date prefixes form a strictly consecutive sequence.

    Returns (sorted_binaries, dates) on success; raises ValueError otherwise.
    Single-binary uploads trivially pass.
    """
    dated = []
    for p in binaries:
        name = os.path.basename(p)
        m = BINARY_FILENAME_RE.match(name)
        d = datetime.strptime(m.group(1), "%Y%m%d").date()
        dated.append((d, p))
    dated.sort(key=lambda x: x[0])
    for i in range(1, len(dated)):
        prev_d, _ = dated[i - 1]
        cur_d, _ = dated[i]
        if cur_d == prev_d:
            raise ValueError(
                f"duplicate binary date {cur_d.isoformat()}: "
                f"{os.path.basename(dated[i-1][1])} vs {os.path.basename(dated[i][1])}"
            )
        if cur_d - prev_d != timedelta(days=1):
            missing = [(prev_d + timedelta(days=n + 1)).isoformat()
                       for n in range((cur_d - prev_d).days - 1)]
            present = ", ".join(d.isoformat() for d, _ in dated)
            raise ValueError(
                f"non-consecutive binaries: {present} — missing "
                + ", ".join(missing)
            )
    return [p for _, p in dated], [d for d, _ in dated]


def _build_channels_payload(merged_channels, dt):
    css_by_key = {
        "SXR1": "--c-sxr1", "SXR2": "--c-sxr2", "SXR3": "--c-sxr3",
        "SXR4": "--c-sxr4", "HXR":  "--c-hxr",  "EUV":  "--c-euv",
    }
    out = []
    for key in ("SXR1", "SXR2", "SXR3", "SXR4", "HXR", "EUV"):
        v = merged_channels[key]
        mins, maxs, bin_size = _downsample_minmax(v, MAX_PLOT_POINTS)
        vmin = float(np.min(v)) if len(v) else 0.0
        vmax = float(np.max(v)) if len(v) else 1.0
        pad = (vmax - vmin) * 0.12 or 0.1
        out.append({
            "key": key,
            "cssvar": css_by_key[key],
            "mins": mins,
            "maxs": maxs,
            "bin_dt": bin_size * dt,
            "min": vmin - pad,
            "max": vmax + pad,
        })
    return out


def _load_dataset(binary_paths, rpt_paths=None, *, require_segments=True):
    """Parse one or more binaries (stitched onto a single timeline), pair the
    .rpt files into segments, and return the front-end JSON payload.

    Raises ValueError on:
      - non-consecutive binary dates
      - .rpt files whose MUSE window doesn't intersect the binary stream
        (the .rpt files don't belong to these binaries)
    """
    if not binary_paths:
        raise ValueError("no binary files in selection")

    sorted_bins, dates = _check_consecutive_dates(list(binary_paths))

    # Parse each binary independently, then stitch.
    per_binary = []
    for p in sorted_bins:
        d = _get_data.get_data_dict(p)
        per_binary.append((p, d))

    dt = float(per_binary[0][1]["dt"])
    samples_per_packet = int(per_binary[0][1]["samples_per_packet"])

    merged_headers = []
    merged_channels = {k: [] for k in
                       ("SXR1", "SXR2", "SXR3", "SXR4", "HXR", "EUV")}
    packet_offsets = []   # (first_packet_idx_in_merged_headers, sample_offset)
    binaries_meta = []
    binary_breaks = []    # times (s) where one binary ends and the next begins
    pkt_cursor = 0
    sample_cursor = 0
    for p, d in per_binary:
        n_pkts = int(d["n_packets"])
        n_samples = n_pkts * samples_per_packet
        packet_offsets.append((pkt_cursor, sample_cursor))
        merged_headers.extend(d["headers"])
        for k in merged_channels:
            merged_channels[k].append(d[k])
        try:
            size = os.path.getsize(p)
        except OSError:
            size = 0
        t_offset = sample_cursor * dt
        T = n_samples * dt
        binaries_meta.append({
            "name": os.path.basename(p),
            "path": p,
            "size": size,
            "t_offset": t_offset,
            "T": T,
            "n_packets": n_pkts,
        })
        pkt_cursor += n_pkts
        sample_cursor += n_samples
        binary_breaks.append(sample_cursor * dt)
    # Drop the trailing "break" — it's just the end of the timeline.
    binary_breaks = binary_breaks[:-1]
    for k in merged_channels:
        merged_channels[k] = np.concatenate(merged_channels[k]) \
            if merged_channels[k] else np.zeros(0)

    n_samples_total = sample_cursor
    T_total = n_samples_total * dt

    use_offsets = packet_offsets if len(per_binary) > 1 else None
    program = _segments.build_segment_program(
        merged_headers, dt, samples_per_packet,
        rpt_paths or [],
        packet_offsets=use_offsets,
    )

    if require_segments and not (rpt_paths or []):
        # Caller will surface a "needs rpts" prompt; we still parsed the
        # channels so we can render the timeline behind the modal.
        pass

    channels = _build_channels_payload(merged_channels, dt)

    primary = binaries_meta[0]
    payload = {
        "T": T_total,
        "N": n_samples_total,
        "n_packets": pkt_cursor,
        "samples_per_packet": samples_per_packet,
        "dt": dt,
        "channels": channels,
        "segments": program["segments"],
        "rpt_errors": program["errors"],
        "binaries": binaries_meta,
        "binary_breaks": binary_breaks,
        "dates": [d.isoformat() for d in dates],
        # Back-compat: the front-end's existing "file" chip still works for
        # single-binary uploads; multi-binary uploads use `binaries` instead.
        "file": {"name": primary["name"], "size": primary["size"],
                 "path": primary["path"]},
    }
    return payload


# ---------------------------------------------------------------------------
# Run state (single active run at a time)
# ---------------------------------------------------------------------------

_LOG = queue.Queue()
_STATE = {
    "status": "idle",          # idle | running | done | error
    "started_at": None,
    "ended_at": None,
    "outputs": [],             # cumulative session files: [{"path": str, "latest": bool}, ...]
    "analysis_dir": None,
    "error": None,
}
_SESSION_PATHS = []  # ordered, deduped list of file paths produced this session
_LOCK = threading.Lock()

# Directory where `asio-gui` was launched — the default seed for file dialogs
# before any picks happen this session (or after wiping persisted state).
_LAUNCH_CWD = os.path.abspath(os.getcwd())

# Persist the last-used dialog directory across runs, keyed by dialog kind.
_LAST_DIR_FILE = os.path.join(
    os.path.expanduser("~"), ".config", "asio_analyze", "gui_state.json"
)
_LAST_DIRS = {}


def _load_last_dirs():
    global _LAST_DIRS
    try:
        with open(_LAST_DIR_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            _LAST_DIRS = {k: v for k, v in data.items() if isinstance(v, str)}
    except (OSError, ValueError):
        _LAST_DIRS = {}


def _save_last_dir(kind, path):
    if not path or not isinstance(path, str):
        return
    directory = path if os.path.isdir(path) else os.path.dirname(path)
    if not directory or not os.path.isdir(directory):
        return
    _LAST_DIRS[kind] = directory
    try:
        os.makedirs(os.path.dirname(_LAST_DIR_FILE), exist_ok=True)
        with open(_LAST_DIR_FILE, "w", encoding="utf-8") as f:
            json.dump(_LAST_DIRS, f)
    except OSError:
        pass


_load_last_dirs()


def _emit(line):
    _LOG.put(line)


class _TeeStream(io.TextIOBase):
    """Stdout/stderr wrapper that mirrors writes to both the real stream
    and the GUI log queue, line by line."""

    def __init__(self, real):
        self._real = real
        self._buf = ""

    def write(self, s):
        if not isinstance(s, str):
            s = str(s)
        try:
            self._real.write(s)
            self._real.flush()
        except Exception:
            pass
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            _emit(line)
        return len(s)

    def flush(self):
        try:
            self._real.flush()
        except Exception:
            pass


def _snapshot_outputs(analysis_dir):
    """Return {abs_path: mtime} for every file under `analysis_dir` (recursive).

    Outputs are split into pdf/, csv/, tex/ subfolders by ``commands._setup_run``,
    so this walks the tree rather than scanning a single level.
    """
    if not analysis_dir or not os.path.isdir(analysis_dir):
        return {}
    snap = {}
    for root, _dirs, files in os.walk(analysis_dir):
        for name in files:
            full = os.path.join(root, name)
            try:
                snap[full] = os.path.getmtime(full)
            except OSError:
                pass
    return snap


def _diff_outputs(before, after):
    """Files that are new or whose mtime changed between `before` and `after`."""
    out = []
    for path, mtime in after.items():
        if path not in before or before[path] != mtime:
            out.append(path)
    return sorted(out)


def _run_command(command, params):
    with _LOCK:
        if _STATE["status"] == "running":
            return False
        # Keep session outputs visible across runs; just clear the "latest" outline.
        prior = [{"path": item["path"], "latest": False} for item in _STATE["outputs"]]
        _STATE.update(
            status="running",
            started_at=time.time(),
            ended_at=None,
            outputs=prior,
            analysis_dir=None,
            error=None,
        )
        while not _LOG.empty():
            try:
                _LOG.get_nowait()
            except queue.Empty:
                break

    def worker():
        directory = params.get("directory")
        anchor = directory if os.path.isdir(directory) else os.path.dirname(directory)
        analysis_dir = params.get("output_dir") or os.path.join(anchor, "analysis")
        _STATE["analysis_dir"] = analysis_dir

        before = _snapshot_outputs(analysis_dir)

        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(old_out)
        sys.stderr = _TeeStream(old_err)
        final_status = "done"
        final_error = None
        try:
            _emit(f"$ asio-analyze {command} {directory}")
            _emit("")
            if command == "default":
                _commands.cmd_default(**params)
            elif command == "background":
                _commands.cmd_background(**params)
            elif command == "ltv":
                _commands.cmd_ltv(**params)
            elif command == "fe55":
                _commands.cmd_fe55(**params)
            elif command == "full":
                _commands.cmd_full(**params)
            else:
                raise ValueError(f"unknown command: {command}")
        except SystemExit as e:
            final_status = "error"
            final_error = f"SystemExit: {e}"
            _emit(f"[error] {final_error}")
        except Exception as e:
            final_status = "error"
            final_error = str(e)
            _emit("[error] " + str(e))
            for line in traceback.format_exc().splitlines():
                _emit(line)
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            after = _snapshot_outputs(analysis_dir)
            new_paths = _diff_outputs(before, after)
            for p in new_paths:
                if p not in _SESSION_PATHS:
                    _SESSION_PATHS.append(p)
            latest = set(new_paths)
            _STATE["outputs"] = [
                {"path": p, "latest": p in latest} for p in _SESSION_PATHS
            ]
            _STATE["error"] = final_error
            _STATE["ended_at"] = time.time()
            if final_status == "done":
                _emit("")
                _emit(f"[done] wrote {len(new_paths)} file(s) to {analysis_dir}")
            # Status flip is the very last step so any poll that sees
            # status == done/error is guaranteed to also see fresh outputs.
            _STATE["status"] = final_status
            _STATE["ended_at"] = time.time()

    threading.Thread(target=worker, daemon=True).start()
    return True


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

_INDEX_HTML = None  # populated at bottom of file


def _seed_dir(start, kind=None):
    """Pick a sensible initial directory for a file dialog.

    Priority: caller-supplied `start` → last directory used for this `kind`
    → cwd at launch → home.
    """
    if start:
        start = os.path.abspath(os.path.expanduser(start))
        if os.path.isdir(start):
            return start
        parent = os.path.dirname(start)
        if parent and os.path.isdir(parent):
            return parent
    if kind:
        last = _LAST_DIRS.get(kind)
        if last and os.path.isdir(last):
            return last
    if os.path.isdir(_LAUNCH_CWD):
        return _LAUNCH_CWD
    return os.path.expanduser("~")


_JXA_DATA_PICK = r"""
ObjC.import('AppKit');
function run(argv) {
  var seed = argv && argv.length ? argv[0] : '';
  var app = $.NSApplication.sharedApplication;
  app.setActivationPolicy(0);  // NSApplicationActivationPolicyRegular
  app.activateIgnoringOtherApps(true);
  var panel = $.NSOpenPanel.openPanel;
  panel.title = 'Select binaries, .rpt files, or a folder';
  panel.canChooseFiles = true;
  panel.canChooseDirectories = true;
  panel.allowsMultipleSelection = true;
  panel.canCreateDirectories = false;
  panel.treatsFilePackagesAsDirectories = true;
  panel.level = 8;
  if (seed && seed.length) {
    panel.directoryURL = $.NSURL.fileURLWithPath(seed);
  }
  app.activateIgnoringOtherApps(true);
  var rc = panel.runModal;
  if (rc != $.NSModalResponseOK) return '';
  var urls = panel.URLs;
  var out = [];
  for (var i = 0; i < urls.count; i++) {
    out.push(ObjC.unwrap(urls.objectAtIndex(i).path));
  }
  return out.join('\n');
}
"""

_JXA_RPTS_PICK = r"""
ObjC.import('AppKit');
function run(argv) {
  var seed = argv && argv.length ? argv[0] : '';
  var app = $.NSApplication.sharedApplication;
  app.setActivationPolicy(0);
  app.activateIgnoringOtherApps(true);
  var panel = $.NSOpenPanel.openPanel;
  panel.title = 'Select .rpt files';
  panel.canChooseFiles = true;
  panel.canChooseDirectories = false;
  panel.allowsMultipleSelection = true;
  panel.allowedFileTypes = ['rpt'];
  panel.level = 8;
  if (seed && seed.length) {
    panel.directoryURL = $.NSURL.fileURLWithPath(seed);
  }
  app.activateIgnoringOtherApps(true);
  var rc = panel.runModal;
  if (rc != $.NSModalResponseOK) return '';
  var urls = panel.URLs;
  var out = [];
  for (var i = 0; i < urls.count; i++) {
    out.push(ObjC.unwrap(urls.objectAtIndex(i).path));
  }
  return out.join('\n');
}
"""

_JXA_OUT_PICK = r"""
ObjC.import('AppKit');
function run(argv) {
  var seed = argv && argv.length ? argv[0] : '';
  var app = $.NSApplication.sharedApplication;
  app.setActivationPolicy(0);  // NSApplicationActivationPolicyRegular
  app.activateIgnoringOtherApps(true);
  var panel = $.NSOpenPanel.openPanel;
  panel.title = 'Select output directory';
  panel.canChooseFiles = false;
  panel.canChooseDirectories = true;
  panel.canCreateDirectories = true;
  panel.allowsMultipleSelection = false;
  panel.level = 8;  // NSModalPanelWindowLevel — above normal windows
  if (seed && seed.length) {
    panel.directoryURL = $.NSURL.fileURLWithPath(seed);
  }
  app.activateIgnoringOtherApps(true);
  var rc = panel.runModal;
  if (rc != $.NSModalResponseOK) return '';
  var url = panel.URLs.objectAtIndex(0);
  return ObjC.unwrap(url.path);
}
"""

# Tkinter fallback runs in a subprocess so it never touches the main thread.
# argv[1] = kind ("data" | "out"), argv[2] = seed directory (may be empty).
_TK_PICK_SCRIPT = r"""
import sys
try:
    import tkinter as tk
    from tkinter import filedialog
except Exception as e:
    sys.stderr.write(str(e))
    sys.exit(2)
kind = sys.argv[1] if len(sys.argv) > 1 else "data"
seed = sys.argv[2] if len(sys.argv) > 2 else ""
root = tk.Tk()
root.withdraw()
try:
    root.attributes("-topmost", True)
    root.lift()
    root.focus_force()
except Exception:
    pass
results = []
if kind == "out":
    r = filedialog.askdirectory(
        title="Select output directory",
        mustexist=False,
        initialdir=seed or None,
    ) or ""
    if r:
        results.append(r)
elif kind == "rpts":
    r = filedialog.askopenfilenames(
        title="Select .rpt files",
        filetypes=[("Report files", "*.rpt"), ("All files", "*.*")],
        initialdir=seed or None,
    )
    results.extend(r or [])
else:
    # "data" / "upload": files first; if cancelled, fall back to folder.
    r = filedialog.askopenfilenames(
        title="Select binaries / .rpt files (cancel for folder)",
        filetypes=[("All files", "*.*")],
        initialdir=seed or None,
    )
    if r:
        results.extend(r)
    else:
        r = filedialog.askdirectory(
            title="Select data folder",
            mustexist=True,
            initialdir=seed or None,
        ) or ""
        if r:
            results.append(r)
try:
    root.destroy()
except Exception:
    pass
sys.stdout.write("\n".join(results))
"""


def _pick_native(kind, start):
    """Spawn an OS-native file picker in a subprocess.

    `kind`:
        "upload" — multi-select files and/or folders (binaries + .rpts).
        "rpts"   — multi-select restricted to .rpt files.
        "out"    — pick a single output directory.
    Returns a list of absolute paths (possibly empty if cancelled).
    """
    seed = _seed_dir(start, kind=kind)

    raw = ""
    if sys.platform == "darwin":
        if kind == "out":
            script = _JXA_OUT_PICK
        elif kind == "rpts":
            script = _JXA_RPTS_PICK
        else:
            script = _JXA_DATA_PICK
        try:
            cp = subprocess.run(
                ["osascript", "-l", "JavaScript", "-e", script, seed],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if cp.returncode == 0:
                raw = cp.stdout or ""
            else:
                raw = None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            raw = None

    if not raw:
        try:
            cp = subprocess.run(
                [sys.executable, "-c", _TK_PICK_SCRIPT, kind, seed],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if cp.returncode == 0:
                raw = cp.stdout or ""
            else:
                raise RuntimeError(
                    cp.stderr.strip() or f"picker exited {cp.returncode}"
                )
        except subprocess.TimeoutExpired:
            raw = ""

    return [p.strip() for p in raw.splitlines() if p.strip()]


def _open_in_default_app(path):
    """Open `path` with the OS's default application for its file type."""
    if sys.platform == "darwin":
        subprocess.Popen(["open", path])
    elif os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", path])


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_a, **_k):
        pass  # silence default logging

    def _send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, body, content_type="text/html; charset=utf-8", status=200):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # ---- GET --------------------------------------------------------------

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._send_text(_INDEX_HTML)
            return
        if u.path == "/favicon.svg":
            try:
                with open(os.path.join(os.path.dirname(__file__), "ASIO_logo.svg"), "rb") as f:
                    self._send_text(f.read(), content_type="image/svg+xml")
            except OSError:
                self._send_text("not found", status=404, content_type="text/plain")
            return
        if u.path == "/api/state":
            qs = parse_qs(u.query)
            since = int(qs.get("since", ["0"])[0])
            # drain log queue without blocking
            lines = []
            while True:
                try:
                    lines.append(_LOG.get_nowait())
                except queue.Empty:
                    break
            self._send_json({
                "status": _STATE["status"],
                "analysis_dir": _STATE["analysis_dir"],
                "outputs": _STATE["outputs"],
                "error": _STATE["error"],
                "lines": lines,
                "version": __version__,
            })
            return
        self._send_text("not found", status=404, content_type="text/plain")

    # ---- POST -------------------------------------------------------------

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            self._send_json({"ok": False, "error": "bad json"}, status=400)
            return

        u = urlparse(self.path)
        if u.path == "/api/open":
            path = (payload.get("path") or "").strip()
            if not path:
                self._send_json({"ok": False, "error": "path required"}, status=400)
                return
            path = os.path.abspath(os.path.expanduser(path))
            analysis_dir = _STATE.get("analysis_dir")
            if not analysis_dir or not path.startswith(
                os.path.abspath(analysis_dir) + os.sep
            ):
                self._send_json({"ok": False, "error": "path outside analysis dir"}, status=403)
                return
            if not os.path.isfile(path):
                self._send_json({"ok": False, "error": "file not found"}, status=404)
                return
            try:
                _open_in_default_app(path)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=500)
                return
            self._send_json({"ok": True})
            return

        if u.path == "/api/load":
            paths = payload.get("paths") or []
            rpt_paths = payload.get("rpt_paths") or []
            require_segments = bool(payload.get("require_segments", True))
            if isinstance(paths, str):
                paths = [paths]
            if not paths:
                self._send_json(
                    {"ok": False, "error": "paths required"}, status=400)
                return
            try:
                binaries, rpts_in_upload, ignored = _classify_upload(paths)
                if not binaries:
                    raise ValueError(
                        "No binary files found in selection "
                        "(expected names like 20260604.0x02B4)"
                    )
                # rpt_paths comes from a follow-up picker; combined with any
                # rpt files included in the initial upload.
                all_rpts = list(rpts_in_upload) + [
                    os.path.abspath(os.path.expanduser(p)) for p in rpt_paths
                ]
                seen = set()
                all_rpts = [x for x in all_rpts
                            if not (x in seen or seen.add(x))]

                # If segment mode requires .rpts and none were provided, ask
                # the front-end to prompt before paying the parse cost.
                if require_segments and not all_rpts:
                    self._send_json({
                        "ok": True,
                        "needs_rpts": True,
                        "binaries": [os.path.basename(p) for p in binaries],
                        "ignored": [os.path.basename(p) for p in ignored],
                    })
                    return

                dataset = _load_dataset(binaries, all_rpts,
                                         require_segments=require_segments)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=500)
                return
            _save_last_dir("upload", binaries[0])
            self._send_json({"ok": True, "data": dataset})
            return

        if u.path == "/api/pick":
            kind = payload.get("kind", "upload")
            if kind == "data":
                kind = "upload"  # back-compat
            if kind not in ("upload", "rpts", "out"):
                self._send_json({"ok": False, "error": "bad kind"}, status=400)
                return
            start = (payload.get("start") or "").strip()
            try:
                chosen = _pick_native(kind, start)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=500)
                return
            if chosen:
                _save_last_dir(kind, chosen[0])
            if kind == "out":
                # legacy single-path callers
                self._send_json({"ok": True,
                                 "path": chosen[0] if chosen else None})
            else:
                self._send_json({"ok": True, "paths": chosen})
            return

        if u.path == "/api/run":
            command = payload.get("command", "default")
            directory = payload.get("directory", "").strip()
            output_dir = (payload.get("output_dir") or "").strip() or None
            note = (payload.get("note") or "").strip() or None
            emit_pdf = bool(payload.get("emit_pdf", False))
            if not directory:
                self._send_json({"ok": False, "error": "data path is required"}, status=400)
                return
            directory = os.path.abspath(os.path.expanduser(directory))
            if not (os.path.isfile(directory) or os.path.isdir(directory)):
                self._send_json({"ok": False, "error": f"path not found: {directory}"}, status=400)
                return
            if output_dir:
                output_dir = os.path.abspath(os.path.expanduser(output_dir))
            params = {"directory": directory, "output_dir": output_dir, "note": note,
                      "emit_pdf": emit_pdf}
            if command != "ltv":
                t0 = payload.get("t0")
                t1 = payload.get("t1")
                if t0 is not None and t1 is not None:
                    try:
                        params["window"] = (float(t0), float(t1))
                    except (TypeError, ValueError):
                        self._send_json(
                            {"ok": False, "error": "t0/t1 must be numeric"},
                            status=400,
                        )
                        return
            if command == "ltv":
                try:
                    lpt_t0 = float(payload.get("lpt_t0"))
                    lpt_t1 = float(payload.get("lpt_t1"))
                    data_t0 = float(payload.get("data_t0"))
                    data_t1 = float(payload.get("data_t1"))
                except (TypeError, ValueError):
                    self._send_json(
                        {"ok": False,
                         "error": "ltv requires lpt_t0, lpt_t1, data_t0, data_t1"},
                        status=400,
                    )
                    return
                params["lpt_window"] = (lpt_t0, lpt_t1)
                params["data_window"] = (data_t0, data_t1)
            started = _run_command(command, params)
            if not started:
                self._send_json({"ok": False, "error": "another run is already in progress"}, status=409)
                return
            self._send_json({"ok": True})
            return

        self._send_json({"ok": False, "error": "not found"}, status=404)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _find_open_port(preferred=8765):
    for port in [preferred] + list(range(8766, 8800)):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", port))
            s.close()
            return port
        except OSError:
            continue
    raise RuntimeError("no free port")


def main(argv=None):
    port = _find_open_port()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"asio-analyze GUI ready at {url}")
    print("Press Ctrl+C to stop.")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()
    return 0


# ---------------------------------------------------------------------------
# Front-end (HTML/CSS/JS) — single string for self-contained deployment
# ---------------------------------------------------------------------------

_INDEX_HTML = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>asio-analyze</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg" />
<style>
:root,[data-theme="dark"]{
  --bg:#1e1e1e; --bg-elevated:#252526; --bg-chrome:#1f1f1f; --bg-inset:#1a1a1a;
  --panel:#2a2a2b; --hover:#2d2d2e; --input:#313131;
  --border:#333334; --border-strong:#454545;
  --text:#cccccc; --text-strong:#e6e6e6; --dim:#969696; --dimmer:#6e6e6e;
  --on-accent:#fff; --accent:#0e639c; --accent-hover:#1177bb; --accent-fg:#fff;
  --focus:#4daafc; --green:#4ec9b0; --red:#f14c4c; --yellow:#cca700;
  --statusbar:#0078d4; --statusbar-fg:#fff;
  --shadow:0 6px 24px rgba(0,0,0,0.45); --selection:rgba(38,121,193,0.35);
  --grid:rgba(255,255,255,0.05); --grid-strong:rgba(255,255,255,0.09);
  --lane-bg:rgba(255,255,255,0.018); --divider:rgba(255,255,255,0.34);
  --c-sxr1:#5BA8E6; --c-sxr2:#3FB6CC; --c-sxr3:#46C39B; --c-sxr4:#86C24E;
  --c-hxr:#E0A33A;  --c-euv:#B07BE0;  --c-relay:#4EC9B0; --c-relay2:#6E8BE6;
  --edge-start:rgba(78,201,176,0.55); --edge-end:rgba(241,76,76,0.55);
  --seg-a:rgba(91,168,230,0.07); --seg-b:rgba(176,123,224,0.07);
  --seg-one:rgba(78,201,176,0.06); --seg-sel:rgba(78,201,176,0.16); --seg-sel-edge:#4EC9B0;
  --z-base:1; --z-overlay:10; --z-panel:20; --z-popover:30; --z-modal:40; --z-toast:50; --z-tooltip:60;
}
[data-theme="light"]{
  --bg:#fff; --bg-elevated:#f8f8f8; --bg-chrome:#f8f8f8; --bg-inset:#f3f3f3;
  --panel:#fff; --hover:#f0f0f0; --input:#fff;
  --border:#e5e5e5; --border-strong:#cecece;
  --text:#3b3b3b; --text-strong:#1f1f1f; --dim:#6b6b6b; --dimmer:#767676;
  --on-accent:#fff; --accent:#005fb8; --accent-hover:#0258a8; --accent-fg:#fff;
  --focus:#0090f1; --green:#098658; --red:#cd3131; --yellow:#b58700;
  --statusbar:#005fb8; --statusbar-fg:#fff;
  --shadow:0 6px 24px rgba(0,0,0,0.14); --selection:rgba(0,95,184,0.18);
  --grid:rgba(0,0,0,0.06); --grid-strong:rgba(0,0,0,0.10);
  --lane-bg:rgba(0,0,0,0.012); --divider:rgba(0,0,0,0.4);
  --c-sxr1:#2F7DBF; --c-sxr2:#1F93A8; --c-sxr3:#1F9E78; --c-sxr4:#4E9E2E;
  --c-hxr:#B97F1E;  --c-euv:#8A53C0;  --c-relay:#098658; --c-relay2:#3A56C0;
  --edge-start:rgba(9,134,88,0.5); --edge-end:rgba(205,49,49,0.5);
  --seg-a:rgba(47,125,191,0.06); --seg-b:rgba(138,83,192,0.06);
  --seg-one:rgba(9,134,88,0.05); --seg-sel:rgba(9,134,88,0.12); --seg-sel-edge:#098658;
}
*{box-sizing:border-box}
::selection{background:var(--selection)}
:focus{outline:none}
:focus-visible{outline:2px solid var(--focus);outline-offset:2px;border-radius:6px}
.input:focus-visible{outline:none;border-color:var(--focus);box-shadow:0 0 0 2px var(--focus)}
.seg:focus-visible,.seg-header:focus-visible{outline-offset:-2px}
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;
  clip:rect(0,0,0,0);white-space:nowrap;border:0}
html,body{margin:0;padding:0;height:100%;background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,"Helvetica Neue",sans-serif;
  font-size:0.8125rem;-webkit-font-smoothing:antialiased;overflow:hidden}
.mono{font-family:"JetBrainsMono Nerd Font","JetBrains Mono",ui-monospace,"SF Mono",Menlo,Consolas,monospace}
#root{height:100vh}
.shell{display:grid;grid-template-rows:44px 1fr 26px;height:100vh}
header.menubar{display:grid;grid-template-columns:1fr auto;align-items:center;
  padding:0 8px 0 14px;background:var(--bg-chrome);border-bottom:1px solid var(--border);gap:16px}
.brand{display:flex;align-items:center;gap:10px;min-width:0}
.brand .logo{width:22px;height:22px;flex-shrink:0;display:block;
  background-image:url("/favicon.svg");background-repeat:no-repeat;
  background-position:center;background-size:contain}
.brand .name{margin:0;font-family:"JetBrainsMono Nerd Font","JetBrains Mono",monospace;
  font-size:0.8125rem;font-weight:600;color:var(--text-strong);letter-spacing:-0.01em;white-space:nowrap}
.brand .sep{color:var(--border-strong)}
.brand .crumb{color:var(--dim);font-size:0.7812rem;white-space:nowrap}
.analysis-file{padding:10px 12px;border-bottom:1px solid var(--border);flex-shrink:0}
.analysis-file .filechip{width:100%;max-width:100%}
.filechip{display:flex;align-items:center;gap:9px;max-width:100%;padding:4px 6px 4px 11px;
  background:var(--panel);border:1px solid var(--border);border-radius:7px;
  color:var(--text-strong);font-family:"JetBrainsMono Nerd Font","JetBrains Mono",monospace;font-size:0.75rem}
.filechip .doticon{width:7px;height:7px;border-radius:50%;background:var(--green);flex-shrink:0}
.filechip .fname{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.filechip .meta{color:var(--dim);font-size:0.6875rem}
.filechip .close{display:grid;place-items:center;width:20px;height:20px;border-radius:5px;
  border:none;background:transparent;color:var(--dim);cursor:pointer}
.filechip .close:hover{background:var(--hover);color:var(--text-strong)}
.titlebar-right{display:flex;align-items:center;gap:6px;justify-self:end}
.icon-btn{display:grid;place-items:center;width:30px;height:30px;border-radius:6px;
  border:1px solid transparent;background:transparent;color:var(--dim);cursor:pointer;
  transition:background .12s,color .12s}
.icon-btn:hover{background:var(--hover);color:var(--text-strong)}
.icon-btn svg{width:16px;height:16px;display:block}
[data-theme="dark"] .sun{display:block}
[data-theme="dark"] .moon{display:none}
[data-theme="light"] .sun{display:none}
[data-theme="light"] .moon{display:block}
.workspace{position:relative;display:flex;min-height:0;min-width:0;overflow:hidden}
.plotcol{display:flex;flex-direction:column;flex:1;min-width:0;min-height:0}
.plotwrap{position:relative;flex:1;min-width:0;min-height:0;background:var(--bg)}
.dropzone-stage{position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:20px;padding:32px;overflow-y:auto}
.proc-mode{display:flex;flex-direction:column;align-items:center;gap:8px}
.proc-mode .pm-label{font-size:0.75rem;color:var(--dim);font-weight:600}
.proc-toggle{display:inline-flex;background:var(--panel);border:1px solid var(--border);
  border-radius:10px;padding:4px;gap:4px}
.proc-toggle button{background:transparent;border:none;color:var(--dim);
  padding:9px 22px;border-radius:7px;font-family:inherit;font-size:0.8125rem;font-weight:600;
  cursor:pointer;transition:background .12s,color .12s;display:flex;flex-direction:column;
  align-items:center;gap:2px;min-width:120px}
.proc-toggle button .sub{font-size:0.6562rem;font-weight:500;color:var(--dimmer);
  letter-spacing:.02em}
.proc-toggle button:hover{color:var(--text-strong)}
.proc-toggle button.active{background:var(--accent);color:var(--accent-fg)}
.proc-toggle button.active .sub{color:var(--accent-fg);opacity:.85}
.dropzone{width:min(1100px,94%);min-height:min(560px,72vh);padding:72px 64px;
  border:1.5px dashed var(--border-strong);
  border-radius:18px;background:var(--bg-elevated);text-align:center;cursor:pointer;
  transition:border-color .15s,background .15s,transform .06s;
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px}
.dropzone:hover{border-color:var(--accent);background:var(--hover)}
.dropzone.drag{border-color:var(--accent);border-style:solid;
  background:color-mix(in srgb,var(--accent) 10%,var(--bg-elevated));transform:scale(1.008)}
.dropzone .glyph{width:84px;height:84px;border-radius:20px;display:grid;place-items:center;
  background:var(--panel);border:1px solid var(--border);color:var(--accent);margin-bottom:18px}
[data-theme="dark"] .dropzone .glyph{color:var(--focus)}
.dropzone .glyph svg{width:40px;height:40px}
.dropzone h2{margin:0;font-size:1.1875rem;font-weight:600;color:var(--text-strong);letter-spacing:-0.01em}
.dropzone .sub{margin:4px 0 0;color:var(--dim);font-size:0.8438rem;line-height:1.5}
.dropzone .browse{margin-top:22px;padding:10px 20px;background:var(--accent);color:var(--accent-fg);
  border:none;border-radius:7px;font-family:inherit;font-size:0.8125rem;font-weight:600;cursor:pointer;
  transition:background .12s}
.dropzone:hover .browse{background:var(--accent-hover)}
.dropzone .hint{margin-top:26px;padding-top:18px;border-top:1px solid var(--border);width:100%;
  color:var(--dimmer);font-size:0.7188rem;font-family:"JetBrainsMono Nerd Font","JetBrains Mono",monospace}
.dropzone .hint b{color:var(--dim);font-weight:600}
.rpt-prompt{cursor:default}
.rpt-prompt:hover{border-color:var(--border-strong);background:var(--bg-elevated);transform:none}
.prompt-actions{display:flex;gap:10px;margin-top:18px;justify-content:center;align-items:stretch}
.prompt-actions .browse{margin-top:0;padding:8px 18px;border-radius:6px;font-size:0.8125rem;line-height:1.2}
.prompt-actions .btn-mini{padding:8px 18px;font-size:0.8125rem;line-height:1.2}
.load-error{max-width:min(900px,90%);padding:12px 16px;
  background:color-mix(in srgb,var(--red) 12%,var(--bg-elevated));
  border:1px solid var(--red);border-radius:8px;color:var(--text);font-size:0.8125rem;line-height:1.5}
.load-error b{color:var(--red);font-weight:600}
.plot-canvas-host{position:absolute;inset:0}
.plot-canvas-host canvas{display:block;width:100%;height:100%}
.plot-overlay{position:absolute;inset:0;pointer-events:none;overflow:hidden}
.zoom-bar{display:flex;align-items:center;gap:8px;padding:8px 12px;
  background:var(--bg-chrome);border-bottom:1px solid var(--border);
  font-family:"JetBrainsMono Nerd Font","JetBrains Mono",monospace;font-size:0.7188rem;flex-shrink:0}
.zoom-bar .zoom-lbl{color:var(--dim);text-transform:uppercase;letter-spacing:0.06em;font-size:0.625rem}
.zoom-bar .zoom-sep,.zoom-bar .zoom-unit{color:var(--dim)}
.zoom-input{width:96px;padding:5px 8px}
.seg{position:absolute;pointer-events:auto;cursor:pointer;transition:background .12s}
.seg:hover{background:color-mix(in srgb,var(--seg-sel) 55%,transparent) !important}
.seg.fill-a{background:var(--seg-a)}
.seg.fill-b{background:var(--seg-b)}
.seg.sel{background:var(--seg-sel) !important;
  box-shadow:inset 0 2px 0 var(--seg-sel-edge),inset 0 -2px 0 var(--seg-sel-edge)}
.seg.sel.sel-first{border-left:2px solid var(--seg-sel-edge)}
.seg.sel.sel-last{border-right:2px solid var(--seg-sel-edge)}
.seg-header{position:absolute;pointer-events:auto;cursor:pointer;
  background:transparent;transition:background .12s;border-radius:6px 6px 0 0}
.seg-header:hover{background:color-mix(in srgb,var(--seg-sel) 28%,transparent)}
.seg-header.sel{background:color-mix(in srgb,var(--seg-sel) 60%,transparent)}
.seg-label{position:absolute;transform:translateX(-50%);display:flex;align-items:center;gap:6px;
  pointer-events:auto;max-width:180px;z-index:var(--z-overlay)}
.seg-label .chip{padding:3px 10px;background:var(--panel);border:1px solid var(--border-strong);
  border-radius:999px;color:var(--text);font-family:"JetBrainsMono Nerd Font","JetBrains Mono",monospace;
  font-size:0.7188rem;font-weight:600;white-space:nowrap;cursor:text;
  overflow:hidden;text-overflow:ellipsis;max-width:170px;
  transition:border-color .12s,color .12s,background .12s}
.seg-label .chip:hover{border-color:var(--accent)}
.seg-label.sel .chip{border-color:var(--seg-sel-edge);color:var(--text-strong);
  background:color-mix(in srgb,var(--seg-sel) 70%,var(--panel))}
.seg-label.notest{pointer-events:none}
.seg-label.notest .chip{background:transparent;border:1px dashed var(--border-strong);
  color:var(--dimmer);font-weight:500}
.seg-label input.chip{outline:none;text-align:center;border-color:var(--focus);
  box-shadow:0 0 0 1px var(--focus)}
.analysis{flex-shrink:0;background:var(--bg-elevated);display:flex;flex-direction:column;overflow:hidden;
  width:360px;height:100%;border-left:1px solid var(--border)}
.analysis-head{display:flex;align-items:center;gap:10px;padding:13px 14px;
  border-bottom:1px solid var(--border);flex-shrink:0}
.analysis-head .ttl{font-size:0.8125rem;font-weight:600;color:var(--text-strong);
  padding:3px 6px;border-radius:5px;
  max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.analysis-head .ttl.editable{cursor:text}
.analysis-head .ttl.editable:hover{background:var(--hover)}
.analysis-head .ttl-edit{font:inherit;font-weight:600;color:var(--text-strong);
  background:var(--input);border:1px solid var(--focus);border-radius:5px;
  padding:3px 8px;outline:none;width:200px;
  box-shadow:0 0 0 1px var(--focus)}
.analysis-head .seg-count{font-family:"JetBrainsMono Nerd Font","JetBrains Mono",monospace;
  font-size:0.6875rem;color:var(--on-accent);background:var(--accent);border-radius:999px;padding:2px 9px}
.analysis-head .close{margin-left:auto;display:grid;place-items:center;width:26px;height:26px;
  border:none;background:transparent;color:var(--dim);border-radius:6px;cursor:pointer}
.analysis-head .close:hover{background:var(--hover);color:var(--text-strong)}
.analysis-body{flex:1;overflow-y:auto;padding:16px 14px}
.a-section{margin-bottom:18px}.a-section:last-child{margin-bottom:0}
.a-section>h3{margin:0 0 10px;font-size:0.75rem;font-weight:600;color:var(--dim)}
.modes{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.mode{position:relative;padding:9px 11px;background:var(--panel);border:1px solid var(--border);
  border-radius:6px;color:var(--text);text-align:left;cursor:pointer;font-family:inherit;
  transition:border-color .12s,background .12s}
.mode:hover{border-color:var(--border-strong);background:var(--hover)}
.mode .k{display:block;font-family:"JetBrainsMono Nerd Font","JetBrains Mono",monospace;
  font-size:0.8125rem;font-weight:600;color:var(--text-strong);margin-bottom:3px}
.mode .d{display:block;font-size:0.6875rem;color:var(--dim);line-height:1.35}
.mode.active{border-color:var(--accent);
  background:color-mix(in srgb,var(--accent) 10%,var(--panel));
  box-shadow:inset 0 0 0 1px var(--accent)}
.mode.active .k{color:var(--accent)}
[data-theme="dark"] .mode.active .k{color:var(--focus)}
.mode.span2{grid-column:span 2}
.field{display:block;margin-bottom:12px}.field:last-child{margin-bottom:0}
.field>label{display:block;font-size:0.7812rem;font-weight:500;color:var(--text);margin-bottom:6px}
.field .sublabel{color:var(--dim);font-weight:400}
.row{display:grid;grid-template-columns:1fr auto;gap:8px}
.input{width:100%;min-width:0;background:var(--input);border:1px solid var(--border-strong);
  border-radius:6px;color:var(--text-strong);padding:8px 11px;
  font-family:"JetBrainsMono Nerd Font","JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;
  font-size:0.7812rem;outline:none;transition:border-color .12s,box-shadow .12s}
.input::placeholder{color:var(--dim)}
.input:focus{border-color:var(--focus);box-shadow:0 0 0 1px var(--focus)}
.btn-mini{background:var(--panel);border:1px solid var(--border-strong);border-radius:6px;
  color:var(--text);padding:0 14px;font-family:inherit;font-size:0.7812rem;font-weight:500;cursor:pointer;
  transition:background .12s,color .12s}
.btn-mini:hover{background:var(--hover);color:var(--text-strong)}
.toggle{display:flex;align-items:center;gap:12px;width:100%;padding:9px 12px;
  background:var(--panel);border:1px solid var(--border);border-radius:6px;color:inherit;font:inherit;
  text-align:left;cursor:pointer;user-select:none;transition:border-color .12s,background .12s}
.toggle:hover{border-color:var(--border-strong);background:var(--hover)}
.toggle .sw{width:30px;height:16px;border-radius:999px;background:var(--border-strong);
  position:relative;flex-shrink:0;transition:background .15s}
.toggle .sw::after{content:'';position:absolute;top:2px;left:2px;width:12px;height:12px;
  border-radius:50%;background:var(--on-accent);box-shadow:0 1px 2px rgba(0,0,0,0.3);transition:left .15s}
.toggle.on .sw{background:var(--accent)}
.toggle.on .sw::after{left:16px}
.toggle .lbl{font-size:0.8125rem;color:var(--text-strong);font-weight:500}
.toggle .hint{color:var(--dim);font-size:0.7188rem;margin-left:auto}
.run{margin-top:16px;width:100%;padding:11px;background:var(--accent);color:var(--accent-fg);
  border:1px solid transparent;border-radius:6px;font-family:inherit;font-size:0.8125rem;font-weight:600;
  letter-spacing:0.02em;cursor:pointer;transition:background .12s,transform .04s}
.run:hover:not(:disabled){background:var(--accent-hover)}
.run:active:not(:disabled){transform:translateY(1px)}
.run:disabled{background:var(--panel);color:var(--dim);border-color:var(--border);cursor:not-allowed}
.ltv-notice{font-size:0.7188rem;color:var(--dim);line-height:1.5;
  padding:9px 11px;background:var(--bg-inset);border:1px solid var(--border);
  border-radius:6px}
.ltv-notice b{color:var(--text-strong);font-weight:600}
.ltv-notice.ltv-error{color:var(--red);border-color:var(--red)}
select.input{appearance:none;-webkit-appearance:none;padding-right:28px;
  background-image:linear-gradient(45deg,transparent 50%,var(--dim) 50%),
                   linear-gradient(135deg,var(--dim) 50%,transparent 50%);
  background-position:calc(100% - 14px) 50%, calc(100% - 9px) 50%;
  background-size:5px 5px;background-repeat:no-repeat;cursor:pointer}
.runresult{margin-top:14px}
.runresult h3{margin:0 0 6px;font-size:0.75rem;font-weight:600;color:var(--dim)}
.runresult h3 + *{margin-top:0}
.runresult .runlog + h3{margin-top:14px}
.runlog{background:var(--bg-inset);border:1px solid var(--border);border-radius:7px;padding:10px 12px;
  font-family:"JetBrainsMono Nerd Font","JetBrains Mono",monospace;font-size:0.7188rem;line-height:1.65;
  color:var(--text);max-height:200px;overflow-y:auto;white-space:pre-wrap;word-break:break-word}
.runlog .cmd{color:var(--focus)}
[data-theme="light"] .runlog .cmd{color:var(--accent)}
.runlog .ok{color:var(--green)}
.runlog .err{color:var(--red)}
.outfiles{display:flex;flex-direction:column;gap:5px;margin-top:10px}
.outfile{display:flex;align-items:center;gap:9px;background:var(--panel);border:1px solid var(--border);
  border-radius:6px;padding:7px 9px;font-family:"JetBrainsMono Nerd Font","JetBrains Mono",monospace;
  font-size:0.7188rem;color:var(--text-strong);cursor:pointer}
.outfile:hover{background:var(--hover)}
.outfile.latest{border-color:color-mix(in srgb,var(--focus) 35%,transparent);
  background:color-mix(in srgb,var(--focus) 5%,transparent)}
.outfile.latest:hover{background:color-mix(in srgb,var(--focus) 9%,transparent)}
.outfile .badge{font-size:0.5625rem;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;
  padding:2px 6px;border-radius:4px;color:var(--on-accent);background:var(--dim);flex-shrink:0}
.outfile.csv .badge{background:var(--green)}
.outfile.pdf .badge{background:var(--red)}
.outfile .nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.status{display:flex;align-items:center;gap:14px;padding:0 12px;background:var(--statusbar);
  color:var(--statusbar-fg);font-size:0.7188rem;transition:background .2s}
.status .seg-i{display:flex;align-items:center;gap:7px}
.status .dot{width:8px;height:8px;border-radius:50%;background:var(--green);
  box-shadow:0 0 0 2px color-mix(in srgb,var(--green) 30%,transparent);
  transition:background .2s,box-shadow .2s}
.status.loaded .dot{background:var(--yellow);
  box-shadow:0 0 0 2px color-mix(in srgb,var(--yellow) 30%,transparent)}
.status.error  .dot{background:var(--red);
  box-shadow:0 0 0 2px color-mix(in srgb,var(--red) 35%,transparent)}
.status.running .dot{animation:blink 1s ease-in-out infinite}
.status.error{background:var(--red)}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.25}}
.status .spacer{flex:1}
.status .v,.status .mono{color:rgba(255,255,255,0.85);
  font-family:"JetBrainsMono Nerd Font","JetBrains Mono",monospace}
.status .label{font-weight:500}
.status svg{width:13px;height:13px}
@media (prefers-reduced-motion: reduce){
  *,*::before,*::after{animation-duration:0.001ms !important;animation-iteration-count:1 !important;
    transition-duration:0.001ms !important}
  .status.running .dot{animation:none;opacity:1}
}
::-webkit-scrollbar{width:12px;height:12px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border-strong);border:3px solid transparent;
  background-clip:padding-box;border-radius:999px}
::-webkit-scrollbar-thumb:hover{background:var(--dim);background-clip:padding-box}
</style>
</head>
<body>
<div id="root">
  <div class="shell">
    <header class="menubar">
      <div class="brand">
        <div class="logo" role="img" aria-label="asio-analyze logo"></div>
        <h1 class="name">asio-analyze</h1>
        <span class="sep">/</span>
        <span class="crumb">Jake's Gooey</span>
      </div>
      <div class="titlebar-right">
        <button class="icon-btn" id="theme-toggle" title="Toggle theme" aria-label="Toggle theme">
          <svg class="sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>
          <svg class="moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
        </button>
      </div>
    </header>
    <main class="workspace" id="workspace" aria-label="Plot and analysis">
      <div class="plotwrap" id="plotwrap">
        <div class="dropzone-stage" id="dropzone-stage">
          <div class="dropzone" id="dropzone">
            <div class="glyph">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
            </div>
            <h2>Drop a binary file to begin</h2>
            <p class="sub">Drag a raw packet capture anywhere here, or click to browse.<br/>asio-analyze parses the channels and HXR10/HXR11 relay states and displays them for you.</p>
            <button class="browse" id="browse-btn" type="button">Browse files</button>
          </div>
        </div>
      </div>
    </main>
    <footer class="status" id="status-bar" role="status" aria-live="polite">
      <div class="seg-i"><span class="dot" aria-hidden="true"></span><span class="label">Ready</span></div>
      <div class="seg-i" id="status-tests"></div>
      <div class="seg-i" id="status-sel"></div>
      <div class="spacer"></div>
      <div class="seg-i" id="status-elapsed"></div>
      <span class="v" id="status-version">asio-analyze</span>
    </footer>
  </div>
</div>
<script>
(function(){
"use strict";

// ---- Tokens & geometry ---------------------------------------------------
const PLOT = { GUTTER:96, PADR:40, labelH:38, xaxisH:32, chMin:64, gap:14, relayH:48, tick:11,
  laneLabelDot:24, laneLabelText:33 };
const MODES = [
  {k:"default",d:"Per-trial stats + voltages"},
  {k:"background",d:"Background test (detrended + EMI)"},
  {k:"fe55",d:"Fe-55 test (raw stats)"},
  {k:"ltv",d:"Light Tightness Verification (LPT vs DATA)"},
  {k:"full",d:"Full report: raw, detrended, FFT, histograms",span:true},
];

// ---- State ---------------------------------------------------------------
const S = {
  theme: localStorage.getItem("asio-theme") ||
    (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"),
  file: null,
  data: null,
  segLabels: {},        // index -> custom label
  selection: null,      // {lo, hi}
  anchor: null,
  viewT0: null,         // null = use 0
  viewT1: null,         // null = use S.data.T
  processMode: localStorage.getItem("asio-process-mode") || "segment",  // "segment" | "whole"
  config: { mode:"default", outputDir:"", note:"", emitPdf:false,
            ltv: { lptIdx: null, dataIdx: null } },
  running: false,
  result: null,         // {lines:[], outputs:[], analysis_dir, error}
  status: "ready",      // ready | loaded | running | done | error
  elapsed: 0,
  version: "",
  // upload flow state
  pendingPaths: null,   // user-selected paths waiting on a load decision
  pendingRpts: null,
  needsRpts: null,      // { binaries: [...], ignored: [...] } when prompting
  loadError: null,
};

document.documentElement.setAttribute("data-theme", S.theme);

// ---- Utilities -----------------------------------------------------------
function $(id){ return document.getElementById(id); }
function el(tag, props, ...children){
  const e = document.createElement(tag);
  if (props) for (const k in props){
    const v = props[k];
    if (k === "class") e.className = v;
    else if (k === "style") Object.assign(e.style, v);
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "html") e.innerHTML = v;
    else if (v === undefined || v === null || v === false) continue;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children) if (c != null) e.append(c.nodeType ? c : document.createTextNode(c));
  return e;
}
function fmtBytes(n){
  if (n < 1024) return n + " B";
  if (n < 1024*1024) return (n/1024).toFixed(1) + " KB";
  if (n < 1024*1024*1024) return (n/(1024*1024)).toFixed(1) + " MB";
  return (n/(1024*1024*1024)).toFixed(2) + " GB";
}
function fmtTime(s){
  const m = Math.floor(s/60), x = s%60;
  return m + ":" + String(x).padStart(2,"0");
}
function cssVar(name){
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
function segmentSummaryText(){
  if (!S.data || !S.data.segments.length) return "";
  const parts = S.data.segments.map((s, i) =>
    "Test " + segLabel(i) + " from " + s.t0.toFixed(2) + "s to " + s.t1.toFixed(2) + "s");
  return "Detected " + S.data.segments.length + " tests: " + parts.join("; ") + ".";
}
function canvasAriaLabel(){
  if (!S.data) return "Empty plot";
  const T = S.data.T.toFixed(1);
  const chKeys = S.data.channels.map(c => c.key).join(", ");
  const nSeg = S.data.segments.length;
  const nBin = (S.data.binaries || []).length;
  return "Plot of " + S.data.channels.length + " channels (" + chKeys + ") over " +
    T + " seconds across " + nBin + " binar" + (nBin === 1 ? "y" : "ies") +
    "; " + nSeg + " test" + (nSeg === 1 ? "" : "s") + " detected.";
}
async function postJSON(url, body){
  const r = await fetch(url, { method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify(body || {}) });
  return r.json();
}

// ---- Theme ---------------------------------------------------------------
function toggleTheme(){
  S.theme = (S.theme === "dark") ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", S.theme);
  localStorage.setItem("asio-theme", S.theme);
  drawPlot();
  drawOverlay();
}
$("theme-toggle").addEventListener("click", toggleTheme);

// ---- File load -----------------------------------------------------------
async function pickAndLoad(){
  const pick = await postJSON("/api/pick", { kind:"upload" });
  if (!pick.ok || !pick.paths || !pick.paths.length) return;
  S.pendingPaths = pick.paths;
  S.pendingRpts = [];
  await runLoad();
}
async function runLoad(){
  const requireSegments = S.processMode === "segment";
  setStatus("running", "Loading…");
  try {
    const res = await postJSON("/api/load", {
      paths: S.pendingPaths || [],
      rpt_paths: S.pendingRpts || [],
      require_segments: requireSegments,
    });
    if (!res.ok) {
      setStatus("error", res.error || "load failed");
      S.loadError = res.error || "load failed";
      render();
      return;
    }
    if (res.needs_rpts) {
      // Segment-mode gate: prompt user to add .rpt files (or go back).
      S.needsRpts = {
        binaries: res.binaries || [],
        ignored: res.ignored || [],
      };
      setStatus("ready", "Pick .rpt files");
      render();
      return;
    }
    S.needsRpts = null;
    S.loadError = null;
    S.data = res.data;
    S.file = res.data.file;
    S.selection = null; S.anchor = null; S.segLabels = {}; S.result = null;
    S.viewT0 = null; S.viewT1 = null;
    S.config.ltv = { lptIdx: null, dataIdx: null };
    setStatus("loaded", "Loaded");
    render();
  } catch (e) {
    setStatus("error", String(e));
  }
}
async function addRpts(){
  const pick = await postJSON("/api/pick", { kind:"rpts" });
  if (!pick.ok || !pick.paths || !pick.paths.length) return;
  S.pendingRpts = (S.pendingRpts || []).concat(pick.paths);
  await runLoad();
}
function cancelLoad(){
  S.pendingPaths = null;
  S.pendingRpts = null;
  S.needsRpts = null;
  S.loadError = null;
  setStatus("ready", "Ready");
  render();
}
function closeFile(){
  S.file = null; S.data = null; S.selection = null; S.anchor = null;
  S.segLabels = {}; S.result = null;
  S.viewT0 = null; S.viewT1 = null;
  S.config.ltv = { lptIdx: null, dataIdx: null };
  S.pendingPaths = null; S.pendingRpts = null; S.needsRpts = null;
  S.loadError = null;
  setStatus("ready", "Ready");
  render();
}

// ---- Status --------------------------------------------------------------
function setStatus(status, label){
  S.status = status;
  const bar = $("status-bar");
  bar.classList.remove("running", "error", "done", "loaded");
  if (status === "running") bar.classList.add("running");
  if (status === "error") bar.classList.add("error");
  if (status === "done") bar.classList.add("done");
  if (status === "loaded") bar.classList.add("loaded");
  bar.querySelector(".label").textContent = label || ({
    ready:"Ready", loaded:"Loaded", running:"Running", done:"Done", error:"Error"
  })[status] || status;
  updateStatusBits();
}
function updateStatusBits(){
  const t = $("status-tests");
  if (S.data){
    t.innerHTML = "";
    t.appendChild(el("span", {class:"mono"}, S.data.segments.length + " tests"));
  } else { t.innerHTML = ""; }

  const s = $("status-sel");
  s.innerHTML = "";
  if (S.data){
    let txt;
    if (!S.selection) txt = "No test selected";
    else if (S.selection.lo === S.selection.hi) txt = segLabel(S.selection.lo) + " selected";
    else txt = segLabel(S.selection.lo) + "–" + segLabel(S.selection.hi) + " selected";
    s.appendChild(el("span", {}, txt));
  }
  $("status-elapsed").textContent = S.running ? ("⏱ " + fmtTime(S.elapsed)) : "";
  $("status-version").textContent = "asio-analyze " + (S.version || "");
}

// ---- Render top ----------------------------------------------------------
function render(){
  renderWorkspace();
  updateStatusBits();
}
function renderFileChip(){
  if (!S.file) return null;
  const bins = (S.data && S.data.binaries) || [];
  let label, meta;
  if (bins.length > 1){
    const first = bins[0].name, last = bins[bins.length - 1].name;
    label = first + " … " + last;
    const total = bins.reduce((a, b) => a + (b.size || 0), 0);
    meta = "· " + bins.length + " binaries · " + fmtBytes(total);
  } else {
    label = S.file.name;
    meta = "· " + fmtBytes(S.file.size);
  }
  return el("div", {class:"filechip"},
    el("span", {class:"doticon", "aria-hidden":"true"}),
    el("span", {class:"fname"}, label),
    el("span", {class:"meta"}, meta),
    el("button", {class:"close", title:"Close file", "aria-label":"Close file",
      onclick: closeFile, html:'<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><line x1="6" y1="6" x2="18" y2="18"/><line x1="18" y1="6" x2="6" y2="18"/></svg>'})
  );
}
function renderZoomBar(){
  const T = (S.data && S.data.T) || 0;
  const bar = el("div", {class:"zoom-bar"});
  bar.appendChild(el("span", {class:"zoom-lbl"}, "View"));
  const mk = (key, val, ph) => {
    const i = el("input", {
      class:"input zoom-input", type:"number", step:"0.1",
      min:"0", max: String(T.toFixed(3)),
      placeholder: ph,
      value: (val != null ? String(val) : ""),
    });
    i.addEventListener("change", () => applyZoom(key, i.value));
    i.addEventListener("keydown", (e) => { if (e.key === "Enter") i.blur(); });
    return i;
  };
  bar.appendChild(mk("viewT0", S.viewT0, "0.0"));
  bar.appendChild(el("span", {class:"zoom-sep"}, "→"));
  bar.appendChild(mk("viewT1", S.viewT1, T.toFixed(1)));
  bar.appendChild(el("span", {class:"zoom-unit"}, "s"));
  const reset = el("button", {class:"btn-mini", type:"button",
    onclick: () => { S.viewT0 = null; S.viewT1 = null; renderWorkspace(); }
  }, "Reset");
  bar.appendChild(reset);
  return bar;
}
function applyZoom(key, raw){
  const T = (S.data && S.data.T) || 0;
  const s = (raw || "").trim();
  if (s === ""){ S[key] = null; }
  else {
    let n = parseFloat(s);
    if (!isFinite(n)) { renderWorkspace(); return; }
    n = Math.max(0, Math.min(T, n));
    S[key] = n;
  }
  const lo = (S.viewT0 != null) ? S.viewT0 : 0;
  const hi = (S.viewT1 != null) ? S.viewT1 : T;
  if (hi - lo < 1e-6){
    if (key === "viewT0") S.viewT0 = null;
    else S.viewT1 = null;
  }
  renderWorkspace();
}
function renderWorkspace(){
  const ws = $("workspace");
  ws.innerHTML = "";
  const col = el("div", {class:"plotcol"});
  ws.appendChild(col);
  const plot = el("div", {class:"plotwrap", id:"plotwrap"});
  if (S.file) col.appendChild(renderZoomBar());
  col.appendChild(plot);
  if (!S.file){
    plot.appendChild(renderDropzone());
    return;
  }
  // canvas + overlay
  const host = el("div", {class:"plot-canvas-host"});
  const c = el("canvas", {id:"plot-canvas", role:"img", "aria-label": canvasAriaLabel()});
  host.appendChild(c);
  // Screen-reader summary of the segment timing, so AT users get an
  // overview without having to step through every segment button.
  host.appendChild(el("div", {class:"sr-only"}, segmentSummaryText()));
  plot.appendChild(host);
  const ov = el("div", {class:"plot-overlay", id:"plot-overlay"});
  plot.appendChild(ov);
  // Handle clicks outside segments to clear selection
  plot.addEventListener("click", (e) => {
    if (e.target === plot || e.target === host || e.target === c || e.target === ov){
      S.selection = null; S.anchor = null;
      renderWorkspace();
      updateStatusBits();
    }
  });
  // LTV always shows the panel (its picks live in the panel itself, not on the plot).
  if (S.processMode === "whole" || S.selection || S.config.mode === "ltv")
    ws.appendChild(renderAnalysisPanel());
  requestAnimationFrame(() => { drawPlot(); drawOverlay(); });
}
function setProcessMode(m){
  if (m !== "segment" && m !== "whole") return;
  if (S.processMode === m) return;
  S.processMode = m;
  localStorage.setItem("asio-process-mode", m);
  S.selection = null; S.anchor = null; S.result = null;
  renderWorkspace(); updateStatusBits();
}
function renderProcessToggle(){
  const wrap = el("div", {class:"proc-mode"});
  wrap.appendChild(el("div", {class:"pm-label"}, "Processing mode"));
  const tg = el("div", {class:"proc-toggle", role:"group"});
  const mk = (key, label, sub) => {
    const b = el("button", {
      type:"button",
      class: (S.processMode === key ? "active" : ""),
      "aria-pressed": S.processMode === key ? "true" : "false",
      "aria-label": label + ", " + sub,
      onclick: (e) => { e.stopPropagation(); setProcessMode(key); }
    },
      el("span", {}, label),
      el("span", {class:"sub"}, sub)
    );
    return b;
  };
  tg.appendChild(mk("segment", "Segment", "per-test analysis"));
  tg.appendChild(mk("whole",   "Whole",   "entire dataset"));
  wrap.appendChild(tg);
  return wrap;
}
function renderDropzone(){
  const stage = el("div", {class:"dropzone-stage", id:"dropzone-stage"});
  stage.appendChild(renderProcessToggle());

  if (S.needsRpts){
    stage.appendChild(renderRptPrompt());
    return stage;
  }

  if (S.loadError){
    stage.appendChild(el("div", {class:"load-error"},
      el("b", {}, "Couldn't load: "), S.loadError));
  }

  const dz = el("div", {class:"dropzone", id:"dropzone"},
    el("div", {class:"glyph", html:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>'}),
    el("h2", {}, "Drop binaries and .rpt files to begin"),
    el("p", {class:"sub", html:"Drag one or more raw packet captures (and any associated .rpt files), or a folder containing them.<br/>asio-analyze stitches consecutive-day binaries onto one timeline and uses each .rpt pair to mark a test segment."}),
    el("button", {class:"browse", type:"button", onclick:(e)=>{ e.stopPropagation(); pickAndLoad(); }}, "Browse files")
  );
  dz.addEventListener("click", pickAndLoad);
  stage.appendChild(dz);
  // drag
  ["dragenter","dragover"].forEach(ev => stage.addEventListener(ev, (e)=>{
    e.preventDefault(); dz.classList.add("drag");
  }));
  ["dragleave","drop"].forEach(ev => stage.addEventListener(ev, (e)=>{
    e.preventDefault(); dz.classList.remove("drag");
  }));
  stage.addEventListener("drop", (e) => {
    pickAndLoad();
  });
  return stage;
}
function renderRptPrompt(){
  const card = el("div", {class:"dropzone rpt-prompt"});
  card.appendChild(el("h2", {}, "Add .rpt files for segment mode"));
  const bins = S.needsRpts.binaries || [];
  card.appendChild(el("p", {class:"sub"},
    "Segment mode needs the GSE .rpt files that mark each test's start and end. ",
    bins.length + " binary file" + (bins.length === 1 ? "" : "s") + " loaded: " +
      bins.join(", ") + "."));
  const row = el("div", {class:"prompt-actions"});
  row.appendChild(el("button", {class:"browse", type:"button",
    onclick: (e) => { e.stopPropagation(); addRpts(); }
  }, "Add .rpt files"));
  row.appendChild(el("button", {class:"btn-mini", type:"button",
    onclick: (e) => { e.stopPropagation(); cancelLoad(); }
  }, "Back"));
  card.appendChild(row);
  return card;
}

// ---- Plot canvas ---------------------------------------------------------
function laneList(){
  if (!S.data) return [];
  return S.data.channels.map(ch => ({kind:"channel", channel:ch}));
}
function plotLayout(){
  const wrap = $("plotwrap");
  const W = wrap.clientWidth, H = wrap.clientHeight;
  const dataLeft = PLOT.GUTTER, dataRight = W - PLOT.PADR;
  const dataTop = PLOT.labelH, dataBottom = H - PLOT.xaxisH;
  const lanes = laneList();
  const chs = lanes.length;
  const innerH = dataBottom - dataTop;
  let chH = Math.max(PLOT.chMin, (innerH - (chs - 1) * PLOT.gap) / chs);
  const rects = [];
  let y = dataTop;
  for (const l of lanes){
    rects.push({lane:l, x:dataLeft, y:y, w:dataRight - dataLeft, h:chH});
    y += chH + PLOT.gap;
  }
  return { W, H, dataLeft, dataRight, dataTop, dataBottom, rects };
}
function viewRange(){
  const T = (S.data && S.data.T) || 0;
  const t0 = (S.viewT0 != null) ? S.viewT0 : 0;
  const t1 = (S.viewT1 != null) ? S.viewT1 : T;
  return { t0, t1, span: Math.max(t1 - t0, 1e-9) };
}
function tToX(t, lay){
  const v = viewRange();
  return lay.dataLeft + ((t - v.t0) / v.span) * (lay.dataRight - lay.dataLeft);
}
function drawPlot(){
  if (!S.data) return;
  const canvas = $("plot-canvas");
  if (!canvas) return;
  const wrap = $("plotwrap");
  const W = wrap.clientWidth, H = wrap.clientHeight;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.width = W + "px"; canvas.style.height = H + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,W,H);

  const lay = plotLayout();
  const grid = cssVar("--grid");
  const gridStrong = cssVar("--grid-strong");
  const dim = cssVar("--dim");
  const dimmer = cssVar("--dimmer");
  const text = cssVar("--text");
  const laneBg = cssVar("--lane-bg");

  ctx.font = PLOT.tick + "px 'JetBrains Mono', ui-monospace, Menlo, monospace";
  ctx.textBaseline = "middle";

  // Per-lane backgrounds, traces, and mid gridlines.
  for (const r of lay.rects){
    ctx.fillStyle = laneBg;
    ctx.fillRect(r.x, r.y, r.w, r.h);
    ctx.strokeStyle = grid;
    ctx.beginPath();
    ctx.moveTo(r.x, r.y + r.h/2); ctx.lineTo(r.x + r.w, r.y + r.h/2);
    ctx.stroke();
    drawChannel(ctx, r);
  }
  // Inter-lane dividers between the 6 channels.
  let prev = null;
  for (let i = 0; i < lay.rects.length; i++){
    const r = lay.rects[i];
    if (prev){
      const yDiv = (prev.y + prev.h + r.y) / 2;
      ctx.strokeStyle = cssVar("--divider");
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, yDiv);
      ctx.lineTo(lay.dataRight, yDiv);
      ctx.stroke();
    }
    prev = r;
  }
  ctx.lineWidth = 1;

  // Binary boundary dividers: faint vertical lines where consecutive-day
  // captures are stitched together.
  const breaks = S.data.binary_breaks || [];
  if (breaks.length){
    ctx.save();
    ctx.strokeStyle = cssVar("--divider");
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    for (const t of breaks){
      const x = tToX(t, lay);
      ctx.beginPath();
      ctx.moveTo(x, lay.dataTop);
      ctx.lineTo(x, lay.dataBottom);
      ctx.stroke();
    }
    ctx.restore();
  }
  // x axis
  ctx.strokeStyle = gridStrong;
  ctx.beginPath();
  ctx.moveTo(lay.dataLeft, lay.dataBottom); ctx.lineTo(lay.dataRight, lay.dataBottom);
  ctx.stroke();
  ctx.fillStyle = dim;
  ctx.textAlign = "center";
  const ticks = 8;
  const v = viewRange();
  for (let i=0; i<=ticks; i++){
    const t = v.t0 + v.span * i / ticks;
    const x = tToX(t, lay);
    ctx.strokeStyle = gridStrong;
    ctx.beginPath(); ctx.moveTo(x, lay.dataBottom); ctx.lineTo(x, lay.dataBottom + 5); ctx.stroke();
    ctx.fillText(t.toFixed(1) + "s", x, lay.dataBottom + 16);
  }
}
function drawChannel(ctx, r){
  const ch = r.lane.channel;
  const color = cssVar(ch.cssvar);
  const dim = cssVar("--dim");
  const text = cssVar("--text");
  const dimmer = cssVar("--dimmer");
  // gutter label
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(PLOT.laneLabelDot, r.y + r.h/2, 3.5, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = text; ctx.textAlign = "left"; ctx.textBaseline = "middle";
  ctx.fillText(ch.key, PLOT.laneLabelText, r.y + r.h/2);
  // (channel y-axis tick labels intentionally omitted — these plots are for
  // shape/quick scan, not reading values; relay ON/OFF labels still drawn.)
  // trace
  const mins = ch.mins, maxs = ch.maxs;
  const n = mins.length;
  if (n === 0) return;
  ctx.save();
  ctx.beginPath();
  ctx.rect(r.x, r.y, r.w, r.h);
  ctx.clip();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.15;
  // map voltage v -> y
  const yOf = (v) => r.y + r.h - ((v - ch.min) / (ch.max - ch.min)) * r.h;
  ctx.beginPath();
  // For each bin, draw vertical line min->max, then connect across bins.
  for (let i=0; i<n; i++){
    const x = r.x + (i / (n-1 || 1)) * r.w;
    const y1 = yOf(mins[i]);
    const y2 = yOf(maxs[i]);
    if (i === 0) ctx.moveTo(x, y2);
    else ctx.lineTo(x, y2);
  }
  for (let i=n-1; i>=0; i--){
    const x = r.x + (i / (n-1 || 1)) * r.w;
    const y1 = yOf(mins[i]);
    ctx.lineTo(x, y1);
  }
  ctx.closePath();
  ctx.fillStyle = color + "33";
  ctx.fill();
  ctx.beginPath();
  for (let i=0; i<n; i++){
    const x = r.x + (i / (n-1 || 1)) * r.w;
    const y2 = yOf(maxs[i]);
    if (i === 0) ctx.moveTo(x, y2);
    else ctx.lineTo(x, y2);
  }
  ctx.stroke();
  ctx.restore();
}
// ---- Overlay (segments, labels, edges) -----------------------------------
function segLabel(i){
  return (S.segLabels[i] !== undefined) ? S.segLabels[i] : S.data.segments[i].label;
}
function drawOverlay(){
  const ov = $("plot-overlay");
  if (!ov || !S.data) return;
  ov.innerHTML = "";
  const lay = plotLayout();

  if (S.processMode === "whole") return;

  // segments
  S.data.segments.forEach((seg, i) => {
    const x0 = tToX(seg.t0, lay), x1 = tToX(seg.t1, lay);
    const sel = S.selection && i >= S.selection.lo && i <= S.selection.hi;
    const fillClass = sel ? "sel" : (i % 2 === 0 ? "fill-a" : "fill-b");
    const segDiv = el("div", {
      class: "seg " + fillClass +
        (sel && i === S.selection.lo ? " sel-first" : "") +
        (sel && i === S.selection.hi ? " sel-last" : ""),
      style: { left: x0 + "px", top: lay.dataTop + "px",
               width: (x1 - x0) + "px", height: (lay.dataBottom - lay.dataTop) + "px" },
      onclick: (e) => { e.stopPropagation(); selectSegment(i, e.shiftKey); }
    });
    ov.appendChild(segDiv);

    // boundary lines (start = green at t0, end = red at t1)
    const startLine = el("div", {style: {
      position:"absolute", left: x0 + "px", top: lay.dataTop + "px",
      width: "1px", height: (lay.dataBottom - lay.dataTop) + "px",
      background: cssVar("--edge-start"), pointerEvents:"none"
    }});
    const endLine = el("div", {style: {
      position:"absolute", left: x1 + "px", top: lay.dataTop + "px",
      width: "1px", height: (lay.dataBottom - lay.dataTop) + "px",
      background: cssVar("--edge-end"), pointerEvents:"none"
    }});
    ov.appendChild(startLine); ov.appendChild(endLine);

    // Header strip above the segment — full-width click target so the user
    // can grab the segment by clicking anywhere in the label row above it,
    // not just on the chip itself. Sits behind the chip so the chip's own
    // click/dblclick handlers (rename) still win.
    const total = S.data.segments.length;
    const headerHit = el("div", {
      class: "seg-header" + (sel ? " sel" : ""),
      role: "button",
      tabindex: "0",
      "data-idx": i,
      "aria-label": "Test " + segLabel(i) + ", " + (i + 1) + " of " + total +
        (sel ? ", selected" : "") +
        ". Enter or Space to select, Shift+Enter to extend, F2 to rename.",
      "aria-pressed": sel ? "true" : "false",
      style: { left: x0 + "px", top: "0px",
               width: (x1 - x0) + "px", height: lay.dataTop + "px" },
      onclick: (e) => { e.stopPropagation(); selectSegment(i, e.shiftKey); },
      onkeydown: (e) => {
        const n = S.data.segments.length;
        if (e.key === "Enter" || e.key === " "){
          e.preventDefault(); e.stopPropagation();
          selectSegment(i, e.shiftKey);
          requestAnimationFrame(() => focusSegmentHeader(i));
        } else if (e.key === "ArrowRight" || e.key === "ArrowLeft"){
          e.preventDefault(); e.stopPropagation();
          const dir = e.key === "ArrowRight" ? 1 : -1;
          const target = Math.max(0, Math.min(n - 1, i + dir));
          selectSegment(target, e.shiftKey);
          requestAnimationFrame(() => focusSegmentHeader(target));
        } else if (e.key === "Home"){
          e.preventDefault(); selectSegment(0, e.shiftKey);
          requestAnimationFrame(() => focusSegmentHeader(0));
        } else if (e.key === "End"){
          e.preventDefault(); selectSegment(n - 1, e.shiftKey);
          requestAnimationFrame(() => focusSegmentHeader(n - 1));
        } else if (e.key === "F2"){
          e.preventDefault(); e.stopPropagation();
          startRenameByIndex(i);
        } else if (e.key === "Escape" && S.selection){
          e.preventDefault();
          S.selection = null; S.anchor = null;
          renderWorkspace(); updateStatusBits();
        }
      },
      ondblclick: (e) => { e.stopPropagation(); startRenameByIndex(i); },
      title: "Click to select · double-click to rename · arrow keys to navigate"
    });
    ov.appendChild(headerHit);

    // label chip — center over the VISIBLE portion of the segment so very
    // long segments (whose true midpoint is off-screen) still get a chip
    // that sits over the part of the segment the user can actually see.
    // Estimate the chip's half-width from its label (≈7px per char + 16px
    // padding + 2px border) rather than using a fixed 90px clamp — the old
    // clamp matched the chip's *max-width*, which forced narrow single-char
    // chips like "1" to sit ~75px to the right of their actual segment when
    // that segment lived near the left edge of the plot.
    const labelTop = Math.max(4, lay.dataTop - 26);
    const labelText = segLabel(i) || "";
    const chipHalf = Math.min(85, Math.max(10, labelText.length * 4 + 9));
    const visX0 = Math.max(x0, lay.dataLeft);
    const visX1 = Math.min(x1, lay.dataRight);
    const visMid = (visX1 >= visX0) ? (visX0 + visX1) / 2 : (x0 + x1) / 2;
    const center = Math.min(
      Math.max(visMid, lay.dataLeft + chipHalf),
      lay.dataRight - chipHalf
    );
    const lblWrap = el("div", {
      class: "seg-label" + (sel ? " sel" : ""),
      "data-idx": i,
      style: { left: center + "px", top: labelTop + "px" }
    });
    const chip = el("div", {class:"chip", title:"Double-click to rename"}, segLabel(i));
    chip.addEventListener("dblclick", (e) => {
      e.stopPropagation();
      startRename(lblWrap, i);
    });
    lblWrap.appendChild(chip);
    ov.appendChild(lblWrap);
  });

}
function startPanelRename(ttlEl, idx){
  const orig = segLabel(idx);
  const parent = ttlEl.parentNode;
  const input = el("input", {class:"ttl-edit", type:"text", value: orig});
  parent.replaceChild(input, ttlEl);
  input.focus(); input.select();
  let done = false;
  const commit = () => {
    if (done) return; done = true;
    const v = input.value.trim();
    if (v === "" || v === S.data.segments[idx].label) delete S.segLabels[idx];
    else S.segLabels[idx] = v;
    renderWorkspace(); updateStatusBits();
  };
  const cancel = () => { if (done) return; done = true; renderWorkspace(); };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter"){ e.preventDefault(); commit(); }
    else if (e.key === "Escape"){ e.preventDefault(); cancel(); }
  });
  input.addEventListener("blur", commit);
}
function focusSegmentHeader(idx){
  const el_ = document.querySelector('.seg-header[data-idx="' + idx + '"]');
  if (el_) el_.focus();
}
function startRenameByIndex(idx){
  // Re-resolve the chip wrapper in the overlay; it has a data-idx attribute.
  const wrap = document.querySelector(`.seg-label[data-idx="${idx}"]`);
  if (wrap) startRename(wrap, idx);
}
function startRename(wrap, idx){
  wrap.innerHTML = "";
  const orig = segLabel(idx);
  const input = el("input", {class:"chip", value: orig, type:"text"});
  wrap.appendChild(input);
  input.focus(); input.select();
  let done = false;
  const commit = () => {
    if (done) return; done = true;
    const v = input.value.trim();
    if (v === "" || v === S.data.segments[idx].label){
      delete S.segLabels[idx];
    } else {
      S.segLabels[idx] = v;
    }
    renderWorkspace();
    updateStatusBits();
  };
  const cancel = () => { if (done) return; done = true; renderWorkspace(); };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); commit(); }
    else if (e.key === "Escape") { e.preventDefault(); cancel(); }
  });
  input.addEventListener("blur", commit);
}
function selectSegment(i, shift){
  if (shift && S.anchor !== null){
    S.selection = { lo: Math.min(S.anchor, i), hi: Math.max(S.anchor, i) };
  } else {
    S.anchor = i;
    S.selection = { lo: i, hi: i };
  }
  renderWorkspace();
  updateStatusBits();
}

// ---- LTV defaults --------------------------------------------------------
function ensureLtvDefaults(){
  const segs = S.data ? S.data.segments : [];
  if (!segs.length) return;
  let { lptIdx, dataIdx } = S.config.ltv;
  if (lptIdx === null || lptIdx < 0 || lptIdx >= segs.length){
    // Seed from current selection (lo), else first segment
    lptIdx = S.selection ? S.selection.lo : 0;
  }
  if (dataIdx === null || dataIdx < 0 || dataIdx >= segs.length){
    dataIdx = S.selection ? S.selection.hi : Math.min(segs.length - 1, lptIdx + 1);
    if (dataIdx <= lptIdx) dataIdx = Math.min(segs.length - 1, lptIdx + 1);
  }
  S.config.ltv.lptIdx = lptIdx;
  S.config.ltv.dataIdx = dataIdx;
}

// ---- Analysis panel ------------------------------------------------------
function renderAnalysisPanel(){
  const whole = S.processMode === "whole";
  const isLtv = S.config.mode === "ltv";
  const sel = S.selection;
  const single = whole ? true : (sel ? sel.lo === sel.hi : true);
  let title;
  if (isLtv){
    title = "LTV analysis";
  } else if (whole){
    title = "Whole dataset";
  } else {
    title = single ? segLabel(sel.lo) : (segLabel(sel.lo) + " – " + segLabel(sel.hi));
  }
  const count = whole || !sel ? 1 : (sel.hi - sel.lo + 1);
  const panel = el("aside", {class:"analysis"});
  const fileChip = renderFileChip();
  if (fileChip){
    panel.appendChild(el("div", {class:"analysis-file"}, fileChip));
  }
  const renameable = single && !whole && !isLtv && sel;
  const ttl = el("div", {
    class: "ttl" + (renameable ? " editable" : ""),
    title: renameable ? "Double-click to rename" : null,
  }, title);
  if (renameable){
    ttl.addEventListener("dblclick", (e) => {
      e.stopPropagation();
      startPanelRename(ttl, sel.lo);
    });
  }
  const head = el("div", {class:"analysis-head"},
    ttl,
    (!whole && !isLtv && sel && count > 1) ? el("div", {class:"seg-count"}, count + " tests") : null,
    (whole || isLtv) ? null : el("button", {class:"close", "aria-label":"Close panel",
      onclick: () => { S.selection = null; S.anchor = null; renderWorkspace(); updateStatusBits(); },
      html:'<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><line x1="6" y1="6" x2="18" y2="18"/><line x1="18" y1="6" x2="6" y2="18"/></svg>'})
  );
  panel.appendChild(head);
  const body = el("div", {class:"analysis-body"});

  // modes
  const sec1 = el("div", {class:"a-section"});
  sec1.appendChild(el("h3", {}, "Analysis"));
  const modes = el("div", {class:"modes"});
  for (const m of MODES){
    const active = S.config.mode === m.k;
    const card = el("button", {
      class: "mode" + (active?" active":"") + (m.span?" span2":""),
      type:"button",
      "aria-pressed": active ? "true" : "false",
      "aria-label": m.k + ": " + m.d,
      onclick: () => { S.config.mode = m.k; S.result = null; renderWorkspace(); }
    },
      el("span", {class:"k"}, m.k),
      el("span", {class:"d"}, m.d),
    );
    modes.appendChild(card);
  }
  sec1.appendChild(modes);
  body.appendChild(sec1);

  // options
  const sec2 = el("div", {class:"a-section"});
  sec2.appendChild(el("h3", {}, "Options"));
  // output dir
  const dirField = el("div", {class:"field"},
    el("label", {for: "opt-output-dir"}, "Output directory"));
  const dirRow = el("div", {class:"row"});
  const dirInput = el("input", {id: "opt-output-dir", name: "outputDir",
    class:"input", type:"text",
    placeholder:"default: <data dir>/analysis", value: S.config.outputDir});
  dirInput.addEventListener("input", (e) => S.config.outputDir = e.target.value);
  dirRow.appendChild(dirInput);
  dirRow.appendChild(el("button", {class:"btn-mini", type:"button",
    "aria-label": "Browse for output directory",
    onclick: async () => {
      const r = await postJSON("/api/pick", { kind:"out", start: S.config.outputDir });
      if (r.ok && r.path){ S.config.outputDir = r.path; renderWorkspace(); }
    }
  }, "Browse"));
  dirField.appendChild(dirRow);
  sec2.appendChild(dirField);
  // note
  const noteField = el("div", {class:"field"},
    el("label", {for: "opt-note"}, "Note ", el("span", {class:"sublabel"}, "· embedded in PDFs")));
  const noteInput = el("input", {id: "opt-note", name: "note",
    class:"input", type:"text", value: S.config.note,
    placeholder:"e.g. room lights off"});
  noteInput.addEventListener("input", (e) => S.config.note = e.target.value);
  noteField.appendChild(noteInput);
  sec2.appendChild(noteField);
  // LTV segment picks (LPT reference + DATA tested)
  if (isLtv){
    const segs = S.data ? S.data.segments : [];
    if (whole){
      sec2.appendChild(el("div", {class:"field"},
        el("div", {class:"ltv-notice"},
          "LTV compares two segments — switch the processing mode to ",
          el("b", {}, "Segment"), " to pick an LPT reference and a DATA segment.")));
    } else if (segs.length < 2){
      sec2.appendChild(el("div", {class:"field"},
        el("div", {class:"ltv-notice"},
          "LTV needs at least two detected segments in the dataset. ",
          "Only ", String(segs.length), " segment(s) were found.")));
    } else {
      ensureLtvDefaults();
      const makeSelect = (id, role, currentIdx, onChange) => {
        const opts = segs.map((_, i) => el("option",
          { value: String(i), selected: i === currentIdx ? "" : null },
          segLabel(i) + " · " + segs[i].t0.toFixed(2) + "–" + segs[i].t1.toFixed(2) + "s"));
        const sel = el("select", { id, class: "input", "aria-label": role }, ...opts);
        sel.addEventListener("change", (e) => onChange(parseInt(e.target.value, 10)));
        return sel;
      };
      const lptField = el("div", {class:"field"},
        el("label", {for:"opt-ltv-lpt"},
          "LPT reference segment ",
          el("span", {class:"sublabel"}, "· earlier segment")));
      lptField.appendChild(makeSelect("opt-ltv-lpt", "LPT reference segment",
        S.config.ltv.lptIdx,
        (i) => { S.config.ltv.lptIdx = i; renderWorkspace(); }));
      sec2.appendChild(lptField);
      const dataField = el("div", {class:"field"},
        el("label", {for:"opt-ltv-data"},
          "DATA segment ",
          el("span", {class:"sublabel"}, "· later segment, tested against LPT")));
      dataField.appendChild(makeSelect("opt-ltv-data", "DATA segment",
        S.config.ltv.dataIdx,
        (i) => { S.config.ltv.dataIdx = i; renderWorkspace(); }));
      sec2.appendChild(dataField);
      if (S.config.ltv.lptIdx !== null &&
          S.config.ltv.dataIdx !== null &&
          S.config.ltv.dataIdx <= S.config.ltv.lptIdx){
        sec2.appendChild(el("div", {class:"field"},
          el("div", {class:"ltv-notice ltv-error"},
            "DATA segment must come after the LPT segment.")));
      }
    }
  }
  // pdf toggle (default only)
  if (S.config.mode === "default"){
    const t = el("button", {
      class:"toggle" + (S.config.emitPdf?" on":""), type:"button",
      role: "switch",
      "aria-checked": S.config.emitPdf ? "true" : "false",
      "aria-label": "Emit PDF (CSV-only otherwise)",
      onclick: () => { S.config.emitPdf = !S.config.emitPdf; renderWorkspace(); }
    },
      el("span", {class:"sw", "aria-hidden":"true"}),
      el("span", {class:"lbl"}, "Emit PDF"),
      el("span", {class:"hint"}, "CSV-only otherwise")
    );
    sec2.appendChild(t);
  }
  body.appendChild(sec2);

  // run button
  let runLabel, runDisabled = S.running;
  if (isLtv){
    const segs = S.data ? S.data.segments : [];
    const lptIdx = S.config.ltv.lptIdx, dataIdx = S.config.ltv.dataIdx;
    const ready = !whole && segs.length >= 2 &&
      lptIdx !== null && dataIdx !== null && dataIdx > lptIdx;
    runDisabled = runDisabled || !ready;
    runLabel = ready
      ? ("Run ltv: LPT " + segLabel(lptIdx) + " → DATA " + segLabel(dataIdx))
      : "Run ltv";
  } else {
    runLabel = "Run " + S.config.mode + " on " +
      (whole ? "whole dataset"
             : (single ? ("test " + segLabel(sel.lo)) : (count + " tests")));
  }
  const runBtn = el("button", {class:"run", disabled: runDisabled || undefined,
    onclick: runAnalysis},
    S.running ? "● Running…" : runLabel);
  body.appendChild(runBtn);

  // run result
  if (S.result){
    const rr = el("div", {class:"runresult"});
    rr.appendChild(el("h3", {}, "Run log"));
    const log = el("div", {class:"runlog", role:"log",
      "aria-live":"polite", "aria-label":"Run output log"});
    for (const line of (S.result.lines || [])){
      const ln = el("div");
      if (line.startsWith("$ ")) ln.className = "cmd";
      else if (line.startsWith("[done]")) ln.className = "ok";
      else if (line.startsWith("[error]")) ln.className = "err";
      ln.textContent = line;
      log.appendChild(ln);
    }
    rr.appendChild(log);
    if (S.result.outputs && S.result.outputs.length){
      rr.appendChild(el("h3", {}, "Output files"));
      const files = el("div", {class:"outfiles"});
      for (const o of S.result.outputs){
        const name = o.path.split("/").pop();
        const ext = (name.split(".").pop() || "").toLowerCase();
        const extClass = (ext === "csv" || ext === "pdf") ? ext : "";
        const cls = "outfile" + (extClass ? " " + extClass : "") + (o.latest ? " latest" : "");
        const row = el("div", {class: cls,
          role: "button",
          tabindex: "0",
          "aria-label": "Open " + name + (o.latest ? " (latest)" : ""),
          onclick: () => postJSON("/api/open", { path: o.path }),
          onkeydown: (e) => {
            if (e.key === "Enter" || e.key === " "){
              e.preventDefault();
              postJSON("/api/open", { path: o.path });
            }
          }},
          el("span", {class:"badge"}, ext || "file"),
          el("span", {class:"nm"}, name)
        );
        files.appendChild(row);
      }
      rr.appendChild(files);
    }
    body.appendChild(rr);
  }

  panel.appendChild(body);
  return panel;
}

// ---- Run wiring (uses existing /api/run + /api/state polling) ------------
async function runAnalysis(){
  if (!S.file) return;
  const whole = S.processMode === "whole";
  const isLtv = S.config.mode === "ltv";
  let scopedNote;
  const body = {
    command: S.config.mode,
    directory: S.file.path,
    output_dir: S.config.outputDir || null,
    // The toggle is only shown for "default"; every other mode is intrinsically
    // a PDF report and the backend defaults to emit_pdf=True for them, so we
    // must explicitly opt-in here (the API handler defaults to False).
    emit_pdf: S.config.mode === "default" ? S.config.emitPdf : true,
  };
  if (isLtv){
    if (whole) return;
    const segs = S.data.segments;
    const { lptIdx, dataIdx } = S.config.ltv;
    if (lptIdx === null || dataIdx === null || dataIdx <= lptIdx) return;
    const lpt = segs[lptIdx], data = segs[dataIdx];
    body.lpt_t0 = lpt.t0; body.lpt_t1 = lpt.t1;
    body.data_t0 = data.t0; body.data_t1 = data.t1;
    scopedNote =
      "LPT: " + segLabel(lptIdx) + " (t=" + lpt.t0.toFixed(1) + "s–" + lpt.t1.toFixed(1) + "s); " +
      "DATA: " + segLabel(dataIdx) + " (t=" + data.t0.toFixed(1) + "s–" + data.t1.toFixed(1) + "s)" +
      (S.config.note ? ("\n" + S.config.note) : "");
  } else {
    if (!whole && !S.selection) return;
    let t0, t1, labels;
    if (whole){
      t0 = 0; t1 = S.data.T;
      labels = "whole dataset";
    } else {
      const sel = S.selection;
      t0 = S.data.segments[sel.lo].t0;
      t1 = S.data.segments[sel.hi].t1;
      labels = (sel.lo === sel.hi)
        ? ("test " + segLabel(sel.lo))
        : ("tests " + segLabel(sel.lo) + "–" + segLabel(sel.hi));
      body.t0 = t0; body.t1 = t1;
    }
    scopedNote =
      "selection: " + labels + " (t=" + t0.toFixed(1) + "s–" + t1.toFixed(1) + "s)" +
      (S.config.note ? ("\n" + S.config.note) : "");
  }
  body.note = scopedNote;

  S.running = true; S.result = { lines: [], outputs: [] };
  setStatus("running", "Running");
  renderWorkspace();
  const runStart = Date.now();
  const tick = setInterval(() => {
    S.elapsed = Math.floor((Date.now() - runStart) / 1000);
    updateStatusBits();
  }, 1000);

  const r = await postJSON("/api/run", body);
  if (!r.ok){
    clearInterval(tick);
    S.running = false;
    setStatus("error", r.error || "run failed");
    S.result.lines.push("[error] " + (r.error || "run failed"));
    renderWorkspace();
    return;
  }

  // Poll /api/state until status != running. Append log lines directly to
  // the existing .runlog node so we don't rebuild the analysis panel on
  // every tick — only structural changes (status flip, outputs landed) call
  // renderWorkspace().
  const poll = setInterval(async () => {
    const st = await (await fetch("/api/state?since=0")).json();
    S.version = st.version || S.version;
    if (st.lines && st.lines.length){
      for (const ln of st.lines) S.result.lines.push(ln);
      appendRunLogLines(st.lines);
    }
    if (st.status !== "running"){
      clearInterval(poll);
      clearInterval(tick);
      S.running = false;
      S.result.outputs = st.outputs || [];
      S.result.analysis_dir = st.analysis_dir;
      if (st.status === "error"){
        setStatus("error", st.error || "error");
      } else {
        setStatus("done", "Done");
      }
      renderWorkspace();
    }
  }, 400);
}
function appendRunLogLines(lines){
  const log = document.querySelector(".runlog");
  if (!log) return;
  for (const line of lines){
    const ln = document.createElement("div");
    if (line.startsWith("$ ")) ln.className = "cmd";
    else if (line.startsWith("[done]")) ln.className = "ok";
    else if (line.startsWith("[error]")) ln.className = "err";
    ln.textContent = line;
    log.appendChild(ln);
  }
  log.scrollTop = log.scrollHeight;
}

// ---- Idle status polling (for version, etc.) -----------------------------
async function bootStatus(){
  try {
    const st = await (await fetch("/api/state?since=0")).json();
    S.version = st.version || "";
    updateStatusBits();
  } catch (e) {}
}

// ---- Init ----------------------------------------------------------------
let _resizeRaf = 0;
window.addEventListener("resize", () => {
  if (!S.data) return;
  if (_resizeRaf) return;
  _resizeRaf = requestAnimationFrame(() => {
    _resizeRaf = 0;
    drawPlot(); drawOverlay();
  });
});
// Page-level drag-and-drop falls through to native picker.
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  if (!S.file) pickAndLoad();
});
bootStatus();
render();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
