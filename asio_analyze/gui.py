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
import socket
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from urllib.parse import urlparse, parse_qs

from . import __version__
from . import commands as _commands


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
    """Return {abs_path: mtime} for every file in `analysis_dir` (non-recursive)."""
    if not analysis_dir or not os.path.isdir(analysis_dir):
        return {}
    snap = {}
    for name in os.listdir(analysis_dir):
        full = os.path.join(analysis_dir, name)
        if os.path.isfile(full):
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
  panel.title = 'Select CSV file or folder';
  panel.canChooseFiles = true;
  panel.canChooseDirectories = true;
  panel.allowsMultipleSelection = false;
  panel.canCreateDirectories = false;
  panel.allowedFileTypes = $.NSArray.arrayWithObject('csv');
  panel.treatsFilePackagesAsDirectories = true;
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
result = ""
if kind == "out":
    result = filedialog.askdirectory(
        title="Select output directory",
        mustexist=False,
        initialdir=seed or None,
    ) or ""
else:
    result = filedialog.askopenfilename(
        title="Select CSV file (cancel to choose a folder)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialdir=seed or None,
    ) or ""
    if not result:
        result = filedialog.askdirectory(
            title="Select data folder",
            mustexist=True,
            initialdir=seed or None,
        ) or ""
try:
    root.destroy()
except Exception:
    pass
sys.stdout.write(result)
"""


def _pick_native(kind, start):
    """Spawn an OS-native file picker in a subprocess and return the chosen
    absolute path, or "" if the user cancelled.

    `kind` is "data" (CSV file or folder, single dialog) or "out" (folder).
    `start` is an optional seed path; the caller's text-field value.
    """
    seed = _seed_dir(start, kind=kind)

    if sys.platform == "darwin":
        script = _JXA_DATA_PICK if kind == "data" else _JXA_OUT_PICK
        try:
            cp = subprocess.run(
                ["osascript", "-l", "JavaScript", "-e", script, seed],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if cp.returncode == 0:
                return (cp.stdout or "").strip()
            # fall through to tk if osascript itself failed
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    try:
        cp = subprocess.run(
            [sys.executable, "-c", _TK_PICK_SCRIPT, kind, seed],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if cp.returncode == 0:
            return (cp.stdout or "").strip()
        raise RuntimeError(cp.stderr.strip() or f"picker exited {cp.returncode}")
    except subprocess.TimeoutExpired:
        return ""


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

        if u.path == "/api/pick":
            kind = payload.get("kind", "data")
            if kind not in ("data", "out"):
                self._send_json({"ok": False, "error": "bad kind"}, status=400)
                return
            start = (payload.get("start") or "").strip()
            try:
                chosen = _pick_native(kind, start)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=500)
                return
            if chosen:
                _save_last_dir(kind, chosen)
            self._send_json({"ok": True, "path": chosen or None})
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
            if command in ("ltv", "full"):
                try:
                    params["sensitivity"] = float(payload.get("sensitivity", 4.0))
                except (TypeError, ValueError):
                    params["sensitivity"] = 4.0
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
<!-- No web fonts: the GUI runs offline on lab boxes. The mono stack prefers
     locally-installed JetBrainsMono Nerd Font (brew install --cask
     font-jetbrains-mono-nerd-font), then plain JetBrains Mono, then the
     platform system mono (SF Mono / Menlo on macOS, Consolas on Windows,
     ui-monospace elsewhere). All fallbacks are excellent. -->
<style>
  /* ============================================================
     asio-analyze — VS Code / GitHub flavored UI
     Light + Dark themes. Chrome only — functionality untouched.
     ============================================================ */
  :root,
  [data-theme="dark"] {
    --bg:            #1e1e1e;   /* editor */
    --bg-elevated:   #252526;   /* sidebar */
    --bg-chrome:     #1f1f1f;   /* title bar */
    --bg-inset:      #1a1a1a;
    --panel:         #2a2a2b;
    --hover:         #2d2d2e;
    --input:         #313131;
    --border:        #333334;
    --border-strong: #454545;
    --text:          #cccccc;
    --text-strong:   #e6e6e6;
    --dim:           #969696;
    --dimmer:        #6e6e6e;
    --on-accent:     #ffffff;
    --accent:        #0e639c;
    --accent-hover:  #1177bb;
    --accent-fg:     #ffffff;
    --focus:         #4daafc;
    --green:         #4ec9b0;
    --red:           #f14c4c;
    --yellow:        #cca700;
    --statusbar:     #0078d4;
    --statusbar-fg:  #ffffff;
    --shadow:        0 6px 24px rgba(0,0,0,0.45);
    --selection:     rgba(38,121,193,0.35);
  }
  [data-theme="light"] {
    --bg:            #ffffff;
    --bg-elevated:   #f8f8f8;
    --bg-chrome:     #f8f8f8;
    --bg-inset:      #f3f3f3;
    --panel:         #ffffff;
    --hover:         #f0f0f0;
    --input:         #ffffff;
    --border:        #e5e5e5;
    --border-strong: #cecece;
    --text:          #3b3b3b;
    --text-strong:   #1f1f1f;
    --dim:           #6b6b6b;
    --dimmer:        #767676;
    --on-accent:     #ffffff;
    --accent:        #005fb8;
    --accent-hover:  #0258a8;
    --accent-fg:     #ffffff;
    --focus:         #0090f1;
    --green:         #098658;
    --red:           #cd3131;
    --yellow:        #b58700;
    --statusbar:     #005fb8;
    --statusbar-fg:  #ffffff;
    --shadow:        0 6px 24px rgba(0,0,0,0.14);
    --selection:     rgba(0,95,184,0.18);
  }

  * { box-sizing: border-box; }
  ::selection { background: var(--selection); }

  html, body {
    margin: 0; padding: 0; height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, "Helvetica Neue", sans-serif;
    font-size: 13px;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
    overflow: hidden;
  }
  .mono { font-family: "JetBrainsMono Nerd Font", "JetBrains Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace; }

  .shell {
    display: grid;
    grid-template-rows: 44px 1fr 24px;
    height: 100vh;
  }

  /* ---------------- title bar ---------------- */
  header {
    display: grid;
    grid-template-columns: auto 1fr auto;
    align-items: center;
    padding: 0 7px 0 14px;
    background: var(--bg-chrome);
    border-bottom: 1px solid var(--border);
    gap: 16px;
  }
  .brand { display: flex; align-items: center; gap: 10px; }
  .brand .logo {
    width: 22px; height: 22px; border-radius: 5px;
    background: var(--accent);
    display: grid; place-items: center;
    color: var(--on-accent); font-weight: 700; font-size: 12px;
    font-family: "JetBrainsMono Nerd Font", "JetBrains Mono", monospace;
    flex-shrink: 0;
  }
  .brand .name {
    margin: 0;
    font-family: "JetBrainsMono Nerd Font", "JetBrains Mono", monospace;
    font-size: 13px; font-weight: 600;
    color: var(--text-strong);
    letter-spacing: -0.01em;
  }
  .brand .sep { color: var(--border-strong); }
  .brand .crumb { color: var(--dim); font-size: 12.5px; }

  .titlebar-right { display: flex; align-items: center; gap: 6px; margin-left: auto; justify-self: end; }
  .icon-btn {
    display: grid; place-items: center;
    width: 30px; height: 30px;
    border-radius: 6px;
    border: 1px solid transparent;
    background: transparent;
    color: var(--dim);
    cursor: pointer;
    transition: background .12s, color .12s;
  }
  .icon-btn:hover { background: var(--hover); color: var(--text-strong); }
  .icon-btn svg { width: 16px; height: 16px; display: block; }
  [data-theme="dark"] .sun { display: block; }
  [data-theme="dark"] .moon { display: none; }
  [data-theme="light"] .sun { display: none; }
  [data-theme="light"] .moon { display: block; }

  /* ---------------- main grid ---------------- */
  main {
    display: grid;
    grid-template-columns: 400px 1fr;
    min-height: 0;
  }
  .panel-left {
    background: var(--bg-elevated);
    border-right: 1px solid var(--border);
    overflow-y: auto;
    padding: 20px;
  }
  .panel-right {
    display: grid;
    grid-template-rows: 1fr minmax(120px, 36%);
    min-height: 0;
    background: var(--bg);
  }

  /* ---------------- sections ---------------- */
  .section { margin-bottom: 18px; }
  .section:last-of-type { margin-bottom: 0; }
  .section > h3 {
    margin: 0 0 10px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--dim);
  }

  /* mode list */
  .modes { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .mode {
    position: relative;
    padding: 11px 12px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    text-align: left;
    cursor: pointer;
    font-family: inherit;
    transition: border-color .12s, background .12s, box-shadow .12s;
  }
  .mode:hover { border-color: var(--border-strong); background: var(--hover); }
  .mode .k {
    display: block;
    font-family: "JetBrainsMono Nerd Font", "JetBrains Mono", monospace;
    font-size: 13.5px; font-weight: 600;
    color: var(--text-strong);
    margin-bottom: 4px;
  }
  .mode .d {
    display: block;
    font-size: 11.5px;
    color: var(--dim);
    line-height: 1.4;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .mode.active {
    border-color: var(--accent);
    background: var(--panel); /* fallback for browsers without color-mix() */
    background: color-mix(in srgb, var(--accent) 10%, var(--panel));
    box-shadow: inset 0 0 0 1px var(--accent);
  }
  .mode.active .k { color: var(--accent); }
  [data-theme="dark"] .mode.active .k { color: var(--focus); }

  /* fields */
  .field { display: block; margin-bottom: 12px; }
  .field:last-child { margin-bottom: 0; }
  .field > label {
    display: block;
    font-size: 12.5px;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 6px;
  }
  .field .sublabel { color: var(--dim); font-weight: 400; }
  .field .req { color: var(--red); font-weight: 600; margin-left: 2px; }
  .field-error {
    margin: 6px 0 0;
    font-size: 12px;
    color: var(--red);
    line-height: 1.4;
  }
  .input[aria-invalid="true"] {
    border-color: var(--red);
    box-shadow: 0 0 0 1px var(--red);
  }
  .row { display: grid; grid-template-columns: 1fr auto; gap: 8px; }

  .input, .input-flat {
    width: 100%;
    min-width: 0;
    background: var(--input);
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    color: var(--text-strong);
    padding: 9px 12px;
    font-family: "JetBrainsMono Nerd Font", "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
    font-size: 13px;
    outline: none;
    transition: border-color .12s, box-shadow .12s;
  }
  .input::placeholder { color: var(--dimmer); }
  .input:focus {
    border-color: var(--focus);
    box-shadow: 0 0 0 1px var(--focus);
  }

  .btn-mini {
    background: var(--panel);
    border: 1px solid var(--border-strong);
    border-radius: 6px;
    color: var(--text);
    padding: 0 16px;
    font-family: inherit;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: background .12s, border-color .12s, color .12s;
  }
  .btn-mini:hover { background: var(--hover); border-color: var(--border-strong); color: var(--text-strong); }

  /* toggle (VS Code style) */
  .toggle {
    display: flex; align-items: center; gap: 12px;
    width: 100%;
    padding: 10px 12px;
    margin-top: 12px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: inherit;
    font: inherit;
    text-align: left;
    cursor: pointer;
    user-select: none;
    transition: border-color .12s, background .12s;
  }
  .toggle:hover { border-color: var(--border-strong); background: var(--hover); }
  .toggle.disabled { opacity: 0.45; cursor: not-allowed; }
  .toggle .sw {
    width: 30px; height: 16px; border-radius: 999px;
    background: var(--border-strong);
    position: relative; flex-shrink: 0;
    transition: background .15s;
  }
  .toggle .sw::after {
    content: '';
    position: absolute; top: 2px; left: 2px;
    width: 12px; height: 12px; border-radius: 50%;
    background: var(--on-accent);
    box-shadow: 0 1px 2px rgba(0,0,0,0.3);
    transition: left .15s;
  }
  .toggle.on .sw { background: var(--accent); }
  .toggle.on .sw::after { left: 16px; }
  .toggle .lbl { font-size: 13.5px; color: var(--text-strong); font-weight: 500; }
  .toggle .hint { color: var(--dim); font-size: 12px; margin-left: auto; }

  /* run button */
  .run {
    margin-top: 20px;
    width: 100%;
    padding: 12px;
    background: var(--accent);
    color: var(--accent-fg);
    border: 1px solid transparent;
    border-radius: 6px;
    font-family: inherit;
    font-size: 13.5px;
    font-weight: 600;
    letter-spacing: 0.02em;
    cursor: pointer;
    transition: background .12s, transform .04s;
  }
  .run:hover:not(:disabled) { background: var(--accent-hover); }
  .run:active:not(:disabled) { transform: translateY(1px); }
  .run:disabled {
    background: var(--panel);
    color: var(--dim);
    border-color: var(--border);
    cursor: not-allowed;
  }

  /* ---------------- log / outputs ---------------- */
  .log-wrap { display: grid; grid-template-rows: 36px 1fr; min-height: 0; }
  .log-head {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 16px;
    background: var(--bg-elevated);
    border-bottom: 1px solid var(--border);
    font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--dim);
  }
  .log-head .right { display: flex; gap: 12px; align-items: center; }
  .log-head .right .tab {
    text-transform: none; letter-spacing: 0; font-weight: 400;
    color: var(--dim); font-family: "JetBrainsMono Nerd Font", "JetBrains Mono", monospace; font-size: 11px;
  }
  .log {
    overflow-y: auto;
    padding: 14px 16px 20px;
    background: var(--bg);
    color: var(--text);
    font-family: "JetBrainsMono Nerd Font", "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
    font-size: 12.5px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    counter-reset: lineno;
  }
  .log .l {
    display: grid; grid-template-columns: 32px 1fr; gap: 14px;
    align-items: baseline;
  }
  .log .l::before {
    counter-increment: lineno;
    content: counter(lineno);
    color: var(--dimmer);
    font-size: 11px; text-align: right;
    user-select: none;
  }
  .log .l.err > span:last-child { color: var(--red); }
  .log .l.cmd > span:last-child { color: var(--focus); font-weight: 500; }
  [data-theme="light"] .log .l.cmd > span:last-child { color: var(--accent); }
  .log .empty {
    color: var(--dimmer);
    text-align: center;
    margin-top: 64px;
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    font-size: 13px;
  }

  /* outputs */
  .outputs {
    background: var(--bg-elevated);
    border-top: 1px solid var(--border);
    padding: 12px 16px 16px;
    overflow-y: auto;
  }
  .outputs h4 {
    margin: 0 0 10px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--dim);
    display: flex; align-items: center; gap: 10px;
  }
  .outputs h4 .count {
    color: var(--text-strong); font-weight: 600;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 999px; padding: 1px 9px;
    font-size: 10.5px; letter-spacing: 0; text-transform: none;
  }
  .outputs h4 .dir {
    color: var(--dim); text-transform: none; letter-spacing: 0;
    font-size: 11px; font-family: "JetBrainsMono Nerd Font", "JetBrains Mono", monospace;
    margin-left: auto; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; max-width: 60%; text-align: right;
  }
  .files { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 6px; }
  .file {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 10px;
    font-family: inherit;
    font-size: 12px;
    color: inherit;
    text-align: left;
    width: 100%;
    display: flex; align-items: center; gap: 10px;
    cursor: pointer;
    transition: border-color .12s, background .12s;
  }
  .file:hover { border-color: var(--accent); background: var(--hover); }
  .file:active { transform: translateY(1px); }
  .file.latest {
    border-color: var(--accent);
    background: var(--hover); /* fallback for browsers without color-mix() */
    background: color-mix(in srgb, var(--accent) 14%, transparent);
  }
  .file .badge {
    font-size: 9px; font-weight: 700; letter-spacing: 0.06em;
    padding: 2px 6px; border-radius: 4px; text-transform: uppercase;
    color: var(--dim); background: var(--bg-inset); border: 1px solid var(--border);
    flex-shrink: 0;
  }
  .file.pdf .badge { color: var(--on-accent); background: var(--red); border-color: transparent; }
  .file.csv .badge { color: var(--on-accent); background: var(--green); border-color: transparent; }
  .file .name {
    overflow: hidden; white-space: nowrap; text-overflow: ellipsis;
    color: var(--text-strong);
    font-family: "JetBrainsMono Nerd Font", "JetBrains Mono", monospace;
  }
  .outputs .empty { color: var(--dimmer); font-size: 12px; }

  /* ---------------- status bar (VS Code) ---------------- */
  .status {
    display: flex; align-items: center; gap: 14px;
    padding: 0 12px;
    background: var(--statusbar);
    color: var(--statusbar-fg);
    font-size: 11.5px;
    transition: background .2s;
  }
  .status .seg { display: flex; align-items: center; gap: 7px; }
  .status .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: rgba(255,255,255,0.85);
  }
  .status.running .dot { animation: blink 1s ease-in-out infinite; }
  .status.error  { background: var(--red); }
  .status.done   .dot { background: var(--on-accent); }
  @keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.25;} }
  .status .spacer { flex: 1; }
  .status .v, .status #elapsed {
    color: rgba(255,255,255,0.85);
    font-family: "JetBrainsMono Nerd Font", "JetBrains Mono", monospace;
  }
  .status .label { font-weight: 500; }
  .status .seg svg { width: 13px; height: 13px; }

  /* focus-visible: keyboard focus ring on every interactive element */
  .mode:focus-visible,
  .icon-btn:focus-visible,
  .btn-mini:focus-visible,
  .toggle:focus-visible,
  .file:focus-visible,
  .run:focus-visible {
    outline: 2px solid var(--focus);
    outline-offset: 2px;
  }
  .input:focus-visible {
    border-color: var(--focus);
    box-shadow: 0 0 0 1px var(--focus);
  }

  /* visually-hidden helper for screen-reader-only announcements */
  .sr-only {
    position: absolute;
    width: 1px; height: 1px;
    padding: 0; margin: -1px;
    overflow: hidden; clip: rect(0,0,0,0);
    white-space: nowrap; border: 0;
  }

  /* ---------------- responsive (narrow windows + coarse pointers) ---------------- */
  @media (max-width: 900px) {
    html, body { overflow: auto; }
    .shell {
      grid-template-rows: 44px auto 24px;
      min-height: 100vh; height: auto;
    }
    main {
      grid-template-columns: 1fr;
      grid-template-rows: auto auto;
    }
    .panel-left {
      border-right: none;
      border-bottom: 1px solid var(--border);
    }
    .panel-right {
      grid-template-rows: minmax(280px, 60vh) minmax(160px, auto);
    }
    .brand .crumb { display: none; }
  }
  @media (pointer: coarse) {
    .icon-btn { width: 44px; height: 44px; }
    .btn-mini { min-height: 44px; padding: 0 18px; }
    .mode { padding: 14px 14px; }
    .toggle { padding: 14px 12px; }
    .run { padding: 16px; }
    .file { padding: 12px 12px; }
  }

  /* reduced motion */
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
      animation-duration: 0.001ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.001ms !important;
      scroll-behavior: auto !important;
    }
    .status.running .dot { animation: none; opacity: 1; }
  }

  /* scrollbars */
  ::-webkit-scrollbar { width: 12px; height: 12px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb {
    background: var(--border-strong);
    border: 3px solid transparent; background-clip: padding-box;
    border-radius: 999px;
  }
  ::-webkit-scrollbar-thumb:hover { background: var(--dim); background-clip: padding-box; }
</style>
</head>
<body>
  <div class="shell">

    <header>
      <div class="brand">
        <span class="logo" aria-hidden="true">A</span>
        <h1 class="name">asio-analyze</h1>
        <span class="sep" aria-hidden="true">/</span>
        <span class="crumb">Jake's Gooey</span>
      </div>
      <div class="titlebar-right">
        <button class="icon-btn" id="theme-toggle" title="Toggle theme" aria-label="Toggle theme">
          <svg class="sun" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/></svg>
          <svg class="moon" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
        </button>
      </div>
    </header>

    <main>

      <!-- LEFT CONFIG PANEL -->
      <aside class="panel-left">

        <div class="section">
          <h3>Mode</h3>
          <div class="modes" id="modes" role="radiogroup" aria-label="Analysis mode">
            <button class="mode active" data-cmd="default" role="radio" aria-checked="true" tabindex="0">
              <span class="k">default</span>
              <span class="d">Per-trial stats + voltages</span>
            </button>
            <button class="mode" data-cmd="background" role="radio" aria-checked="false" tabindex="-1">
              <span class="k">background</span>
              <span class="d">Detrended + EMI-filtered</span>
            </button>
            <button class="mode" data-cmd="fe55" role="radio" aria-checked="false" tabindex="-1">
              <span class="k">fe55</span>
              <span class="d">Raw Fe-55 stats</span>
            </button>
            <button class="mode" data-cmd="ltv" role="radio" aria-checked="false" tabindex="-1">
              <span class="k">ltv</span>
              <span class="d">Light tightness pass/fail</span>
            </button>
            <button class="mode" data-cmd="full" role="radio" aria-checked="false" tabindex="-1" style="grid-column: span 2;">
              <span class="k">full</span>
              <span class="d">Everything: raw, detrended, FFT, histograms, LTV</span>
            </button>
          </div>
        </div>

        <div class="section">
          <h3>Input</h3>
          <div class="field">
            <label for="data-path">
              Data path
              <span class="req" aria-hidden="true">*</span>
              <span class="sublabel">· file or directory</span>
            </label>
            <div class="row">
              <input id="data-path" class="input" type="text"
                     placeholder="/path/to/csv-or-folder" spellcheck="false" autocomplete="off"
                     required aria-required="true" aria-describedby="data-path-error"/>
              <button class="btn-mini" data-browse="data">Browse</button>
            </div>
            <p id="data-path-error" class="field-error" role="alert" hidden></p>
          </div>
        </div>

        <div class="section">
          <h3>Options</h3>

          <div class="field">
            <label for="out-dir">Output directory <span class="sublabel">· optional</span></label>
            <div class="row">
              <input id="out-dir" class="input" type="text"
                     placeholder="default: &lt;data dir&gt;/analysis" spellcheck="false" autocomplete="off"/>
              <button class="btn-mini" data-browse="out">Browse</button>
            </div>
          </div>

          <div class="field">
            <label for="note">Note <span class="sublabel">· embedded in PDFs</span></label>
            <input id="note" class="input" type="text" placeholder="optional free-text note" autocomplete="off"/>
          </div>

          <div class="field" id="sensitivity-field" hidden>
            <label for="sensitivity">LTV sensitivity <span class="sublabel">· z-score threshold</span></label>
            <input id="sensitivity" class="input" type="number" step="0.1" min="0" value="4.0" autocomplete="off"/>
          </div>

          <button type="button" class="toggle" id="pdf-toggle" role="switch" aria-checked="false">
            <span class="sw" aria-hidden="true"></span>
            <span class="lbl">Emit PDF</span>
          </button>
        </div>

        <button class="run" id="run">Run Analysis</button>

      </aside>

      <!-- RIGHT: LOG + OUTPUTS -->
      <section class="panel-right">
        <div class="log-wrap">
          <div class="log-head">
            <span>Run Log</span>
            <div class="right">
              <span class="tab" id="elapsed-mirror"></span>
              <button class="btn-mini" id="copy-log" aria-label="Copy run log to clipboard" title="Copy log">Copy</button>
              <button class="btn-mini" id="clear-log" aria-label="Clear run log" title="Clear log">Clear</button>
            </div>
          </div>
          <div class="log" id="log" role="log" aria-live="polite" aria-atomic="false" aria-label="Run log">
            <div class="empty">No run yet. Pick a mode, point at your data, and press Run.</div>
          </div>
        </div>

        <div class="outputs" id="outputs-pane">
          <h4>
            <span>Outputs</span>
            <span class="count" id="out-count">0 files</span>
            <span class="dir" id="out-dir-label"></span>
          </h4>
          <div class="files" id="outputs">
            <div class="empty">Files written by the run will appear here.</div>
          </div>
        </div>
      </section>

    </main>

    <!-- STATUS BAR -->
    <div class="status idle" id="status" role="status" aria-live="polite" aria-atomic="true">
      <div class="seg">
        <span class="dot" aria-hidden="true"></span>
        <span class="label">Ready</span>
      </div>
      <div class="seg">
        <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        <span id="elapsed" aria-label="Elapsed time">—</span>
      </div>
      <div class="spacer"></div>
      <div class="seg">
        <span class="v" id="ver"></span>
      </div>
    </div>

  </div>

<!-- ============ THEME TOGGLE (presentational only) ============ -->
<script>
  (function(){
    const root = document.documentElement;
    const saved = localStorage.getItem('asio-theme');
    if (saved === 'light' || saved === 'dark') {
      root.setAttribute('data-theme', saved);
    } else {
      const prefersLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
      root.setAttribute('data-theme', prefersLight ? 'light' : 'dark');
    }
    if (window.matchMedia) {
      window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', (e) => {
        if (localStorage.getItem('asio-theme')) return;
        root.setAttribute('data-theme', e.matches ? 'light' : 'dark');
      });
    }
    document.getElementById('theme-toggle').addEventListener('click', () => {
      const next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
      root.setAttribute('data-theme', next);
      localStorage.setItem('asio-theme', next);
    });
  })();
</script>

<!-- ============ FUNCTIONAL SCRIPT — VERBATIM FROM gui.py (do not edit) ============ -->
<script>
  // ---------- state ----------
  const state = {
    command: 'default',
    running: false,
    runStartedAt: null,
    elapsedTimer: null,
  };

  // ---------- mode chips ----------
  const modeBtns = Array.from(document.querySelectorAll('#modes .mode'));
  function selectMode(btn, opts) {
    modeBtns.forEach(b => {
      const on = b === btn;
      b.classList.toggle('active', on);
      b.setAttribute('aria-checked', on ? 'true' : 'false');
      b.setAttribute('tabindex', on ? '0' : '-1');
    });
    state.command = btn.dataset.cmd;
    const sensField = document.getElementById('sensitivity-field');
    if (sensField) {
      sensField.hidden = !(state.command === 'ltv' || state.command === 'full');
    }
    if (opts && opts.focus) btn.focus();
  }
  modeBtns.forEach((btn, i) => {
    btn.addEventListener('click', () => selectMode(btn));
    btn.addEventListener('keydown', (e) => {
      let next = null;
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = modeBtns[(i + 1) % modeBtns.length];
      else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') next = modeBtns[(i - 1 + modeBtns.length) % modeBtns.length];
      else if (e.key === 'Home') next = modeBtns[0];
      else if (e.key === 'End') next = modeBtns[modeBtns.length - 1];
      if (next) { e.preventDefault(); selectMode(next, { focus: true }); }
    });
  });

  // ---------- pdf toggle ----------
  const pdfToggle = document.getElementById('pdf-toggle');
  pdfToggle.addEventListener('click', () => {
    if (pdfToggle.classList.contains('disabled')) return;
    const on = pdfToggle.classList.toggle('on');
    pdfToggle.setAttribute('aria-checked', on ? 'true' : 'false');
  });

  // ---------- browse (native OS picker) ----------
  async function browse(kind) {
    const seedEl = kind === 'data'
      ? document.getElementById('data-path')
      : document.getElementById('out-dir');
    try {
      const r = await fetch('/api/pick', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ kind, start: seedEl.value.trim() }),
      });
      const data = await r.json();
      if (data.ok && data.path) seedEl.value = data.path;
    } catch (e) { /* user-cancel or transient — leave field untouched */ }
  }
  document.querySelectorAll('[data-browse]').forEach(b => {
    b.addEventListener('click', () => browse(b.dataset.browse));
  });

  // escapeHtml is still used by renderOutputs
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
    }[c]));
  }

  // ---------- run ----------
  const runBtn = document.getElementById('run');
  runBtn.addEventListener('click', runAnalysis);

  async function runAnalysis() {
    if (state.running) return;
    const directory = document.getElementById('data-path').value.trim();
    if (!directory) {
      showFieldError('data-path', 'Data path is required. Pick a CSV file or a folder of CSVs.');
      return;
    }
    clearFieldError('data-path');
    const payload = {
      command: state.command,
      directory,
      output_dir: document.getElementById('out-dir').value.trim(),
      note: document.getElementById('note').value.trim(),
      emit_pdf: pdfToggle.classList.contains('on'),
    };
    if (state.command === 'ltv' || state.command === 'full') {
      const sensRaw = document.getElementById('sensitivity').value.trim();
      const sensNum = parseFloat(sensRaw);
      if (Number.isFinite(sensNum) && sensNum > 0) {
        payload.sensitivity = sensNum;
      }
    }
    clearLog();
    setStatus('running', 'Running');
    state.running = true;
    runBtn.disabled = true;
    runBtn.setAttribute('aria-busy', 'true');
    runBtn.textContent = '● Running...';
    state.runStartedAt = Date.now();
    if (state.elapsedTimer) clearInterval(state.elapsedTimer);
    state.elapsedTimer = setInterval(updateElapsed, 100);
    updateElapsed();
    pokePoll();

    try {
      const r = await fetch('/api/run', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(payload),
      });
      const data = await r.json();
      if (!data.ok) {
        appendLine('[error] ' + (data.error || 'unknown'), 'err');
        endRun('error', 'Error');
      }
    } catch (err) {
      appendLine('[error] ' + err.message, 'err');
      endRun('error', 'Error');
    }
  }

  function showFieldError(id, message) {
    const el = document.getElementById(id);
    const err = document.getElementById(id + '-error');
    el.setAttribute('aria-invalid', 'true');
    if (err) {
      err.textContent = message;
      err.hidden = false;
    }
    el.focus();
  }

  function clearFieldError(id) {
    const el = document.getElementById(id);
    const err = document.getElementById(id + '-error');
    el.removeAttribute('aria-invalid');
    if (err) {
      err.textContent = '';
      err.hidden = true;
    }
  }

  // Clear validation error as soon as the user starts typing.
  document.getElementById('data-path').addEventListener('input', () => {
    if (document.getElementById('data-path').getAttribute('aria-invalid') === 'true') {
      clearFieldError('data-path');
    }
  });

  function endRun(s, label) {
    state.running = false;
    runBtn.disabled = false;
    runBtn.removeAttribute('aria-busy');
    runBtn.textContent = 'Run Analysis';
    setStatus(s, label);
    if (state.elapsedTimer) { clearInterval(state.elapsedTimer); state.elapsedTimer = null; }
  }

  function updateElapsed() {
    if (!state.runStartedAt) return;
    const dt = (Date.now() - state.runStartedAt) / 1000;
    document.getElementById('elapsed').textContent = dt.toFixed(1) + 's';
  }

  function setStatus(cls, label) {
    const s = document.getElementById('status');
    s.className = 'status ' + cls;
    s.querySelector('.label').textContent = label;
  }

  // ---------- log ----------
  const logEl = document.getElementById('log');
  let logHasContent = false;

  function clearLog() {
    logEl.innerHTML = '';
    logHasContent = false;
  }

  function appendLine(text, kind) {
    if (!logHasContent) { logEl.innerHTML = ''; logHasContent = true; }
    const row = document.createElement('div');
    row.className = 'l' + (kind ? ' ' + kind : '');
    const span = document.createElement('span');
    span.textContent = text === '' ? ' ' : text;
    row.appendChild(span);
    logEl.appendChild(row);
    logEl.scrollTop = logEl.scrollHeight;
  }
  document.getElementById('clear-log').addEventListener('click', () => {
    logEl.innerHTML = '<div class="empty">Log cleared.</div>';
    logHasContent = false;
  });
  const copyBtn = document.getElementById('copy-log');
  const copyBtnDefaultLabel = copyBtn.textContent;
  let copyBtnTimer = null;
  copyBtn.addEventListener('click', async () => {
    const text = logHasContent
      ? Array.from(logEl.querySelectorAll(':scope > .l'))
          .map(el => el.textContent)
          .join('\n')
      : '';
    let ok = false;
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
        ok = true;
      } else {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.setAttribute('readonly', '');
        ta.style.position = 'absolute';
        ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.select();
        ok = document.execCommand('copy');
        document.body.removeChild(ta);
      }
    } catch (_e) { ok = false; }
    copyBtn.textContent = ok ? 'Copied' : 'Copy failed';
    copyBtn.setAttribute('aria-label', ok ? 'Run log copied to clipboard' : 'Copy failed');
    if (copyBtnTimer) clearTimeout(copyBtnTimer);
    copyBtnTimer = setTimeout(() => {
      copyBtn.textContent = copyBtnDefaultLabel;
      copyBtn.setAttribute('aria-label', 'Copy run log to clipboard');
      copyBtnTimer = null;
    }, 1500);
  });

  // ---------- outputs ----------
  async function openFile(path) {
    try {
      const r = await fetch('/api/open', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ path }),
      });
      const data = await r.json();
      if (!data.ok) appendLine('[error] could not open ' + path + ': ' + (data.error || 'unknown'), 'err');
    } catch (e) {
      appendLine('[error] could not open ' + path + ': ' + e.message, 'err');
    }
  }

  function renderOutputs(files, dir) {
    const wrap = document.getElementById('outputs');
    const count = document.getElementById('out-count');
    const dirLabel = document.getElementById('out-dir-label');
    dirLabel.textContent = dir || '';
    // Normalize: accept either array of strings (legacy) or array of {path, latest}.
    const items = (files || []).map(f =>
      (typeof f === 'string') ? { path: f, latest: false } : f
    );
    const latestCount = items.filter(i => i.latest).length;
    const total = items.length;
    count.textContent = total + ' file' + (total === 1 ? '' : 's') +
                       (latestCount ? ' · ' + latestCount + ' new' : '');
    if (!total) {
      wrap.innerHTML = '<div class="empty">Files written this session will appear here. The latest run is outlined.</div>';
      return;
    }
    wrap.innerHTML = '';
    // Latest files first so they're easy to spot.
    items.sort((a, b) => (b.latest === true) - (a.latest === true));
    for (const item of items) {
      const f = item.path;
      const base = f.split('/').pop();
      const lower = base.toLowerCase();
      const ext = lower.endsWith('.pdf') ? 'pdf'
                : lower.endsWith('.csv') ? 'csv'
                : lower.split('.').pop();
      const el = document.createElement('button');
      el.type = 'button';
      el.className = 'file ' + ext + (item.latest ? ' latest' : '');
      el.title = 'Open ' + f;
      el.setAttribute('aria-label', 'Open ' + base);
      el.addEventListener('click', () => openFile(f));
      el.innerHTML =
        '<span class="badge" aria-hidden="true">' + escapeHtml(ext) + '</span>' +
        '<span class="name" title="' + escapeHtml(f) + '">' + escapeHtml(base) + '</span>';
      wrap.appendChild(el);
    }
  }

  // ---------- poll loop ----------
  // 250ms while a run is in flight; 2000ms when idle; suspended when the tab
  // is hidden. The server owns all run state, so a slow idle cadence is fine —
  // we only need the log/version to refresh promptly when something is happening.
  let pollTimer = null;

  async function poll() {
    try {
      const r = await fetch('/api/state');
      const data = await r.json();
      document.getElementById('ver').textContent = 'v' + data.version;
      for (const line of (data.lines || [])) {
        let kind = '';
        if (line.startsWith('$ ')) kind = 'cmd';
        else if (line.startsWith('[error]') || line.toLowerCase().startsWith('traceback')) kind = 'err';
        appendLine(line, kind);
      }
      if (state.running) {
        if (data.status === 'done') {
          renderOutputs(data.outputs, data.analysis_dir);
          endRun('done', 'Done');
        } else if (data.status === 'error') {
          renderOutputs(data.outputs, data.analysis_dir);
          endRun('error', 'Error');
        }
      }
    } catch (e) { /* ignore transient */ }
  }

  function schedulePoll() {
    if (pollTimer || document.hidden) return;
    const delay = state.running ? 250 : 2000;
    pollTimer = setTimeout(async () => {
      pollTimer = null;
      await poll();
      schedulePoll();
    }, delay);
  }

  function pokePoll() {
    // Cancel any pending wait and poll immediately, then resume scheduling.
    if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
    poll().then(schedulePoll);
  }

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
    } else {
      pokePoll();
    }
  });

  // Initial fetch (no wait) then start the scheduler.
  pokePoll();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
