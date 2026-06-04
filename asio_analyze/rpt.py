"""Parse Galaxy GSE `.rpt` files and convert them into test segment boundaries.

A test produces two reports:
    *_asio_start_test*.rpt   marks the START
    *_asio_end_test*.rpt     marks the END

Each contains telemetry snapshots taken at the moment the proc ran:
    ASIO_TLM_PKT.ASIO_TIME     ms since instrument boot (resets on reboot)
    ASIO_TLM_PKT.TIME_SECONDS  MUSE seconds since 1958-01-01 (monotonic)
    ASIO_TLM_PKT.TIME_SUBSECS  fractional MUSE time

The last line of each .rpt also embeds the EGSE wall-clock time, which we use
to pair starts with ends when multiple tests live in one binary.

Algorithm is a direct port of cells `b0bb91d0` and `7781a260` in
`/Users/evanwilliams/Desktop/segmentizing/flipsample.ipynb`.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta

import numpy as np


_RPT_FIELDS = ("ASIO_TIME", "TIME_SECONDS", "TIME_SUBSECS")
_PROC_COMPLETE_RE = re.compile(
    r"(\d{4}/\d{2}/\d{2}-\d{2}:\d{2}:\d{2})\s+\S+\.rpt proc complete"
)
_TS_FMT = "%Y/%m/%d-%H:%M:%S"

# Tolerance for pairing a start.rpt with the next end.rpt by EGSE wall-clock.
PAIR_WINDOW = timedelta(minutes=10)


def parse_rpt(path):
    """Parse one `.rpt` file.

    Returns a dict with keys:
        path, name, kind ("start" | "end"),
        asio_time_ms, muse_sec, muse_subsec,
        wall_clock (datetime from the trailing "proc complete" line).
    """
    name = os.path.basename(path)
    lower = name.lower()
    if "asio_start_test" in lower:
        kind = "start"
    elif "asio_end_test" in lower:
        kind = "end"
    else:
        raise ValueError(
            f"{name}: filename must contain 'asio_start_test' or 'asio_end_test'"
        )

    with open(path, "r") as f:
        text = f.read()

    fields = {}
    for field in _RPT_FIELDS:
        m = re.search(rf"ASIO_TLM_PKT\.{field}\s*=\s*(\d+)", text)
        if m is None:
            raise ValueError(f"{name}: missing field ASIO_TLM_PKT.{field}")
        fields[field] = int(m.group(1))

    m = _PROC_COMPLETE_RE.search(text)
    if m is None:
        raise ValueError(f"{name}: missing 'proc complete' timestamp line")
    wall = datetime.strptime(m.group(1), _TS_FMT)

    return {
        "path": os.path.abspath(path),
        "name": name,
        "kind": kind,
        "asio_time_ms": fields["ASIO_TIME"],
        "muse_sec": fields["TIME_SECONDS"],
        "muse_subsec": fields["TIME_SUBSECS"],
        "wall_clock": wall,
    }


def pair_rpts(paths):
    """Pair start/end .rpt files by their EGSE wall-clock timestamps.

    Returns (pairs, errors):
        pairs  - list of {"start": <parsed>, "end": <parsed>}
        errors - list of {"path", "reason"} for unmatched / malformed files.

    Pairing rule: sort all files by wall_clock, walk in order. Each start
    binds to the next end whose wall_clock is later but within `PAIR_WINDOW`.
    """
    parsed = []
    errors = []
    for p in paths:
        try:
            parsed.append(parse_rpt(p))
        except (OSError, ValueError) as e:
            errors.append({"path": p, "reason": str(e)})

    parsed.sort(key=lambda r: r["wall_clock"])

    pairs = []
    pending_start = None
    for r in parsed:
        if r["kind"] == "start":
            if pending_start is not None:
                errors.append({
                    "path": pending_start["path"],
                    "reason": "no end .rpt within "
                              f"{int(PAIR_WINDOW.total_seconds() // 60)} min of start",
                })
            pending_start = r
        else:  # "end"
            if pending_start is None:
                errors.append({
                    "path": r["path"],
                    "reason": "end .rpt with no preceding start",
                })
                continue
            if r["wall_clock"] - pending_start["wall_clock"] > PAIR_WINDOW:
                errors.append({
                    "path": pending_start["path"],
                    "reason": "no end .rpt within "
                              f"{int(PAIR_WINDOW.total_seconds() // 60)} min of start",
                })
                errors.append({
                    "path": r["path"],
                    "reason": "end .rpt too far from preceding start",
                })
                pending_start = None
                continue
            pairs.append({"start": pending_start, "end": r})
            pending_start = None

    if pending_start is not None:
        errors.append({
            "path": pending_start["path"],
            "reason": "no matching end .rpt",
        })

    return pairs, errors


def segments_from_rpts(headers, pairs, samples_per_packet, dt,
                       packet_offsets=None):
    """Convert (start, end) .rpt pairs into segment time windows.

    Args:
        headers: list of per-packet header dicts with keys
            'muse_sec', 'asio_time_ms', 'first_dp_ms'. May be the merged
            list across multiple stitched binaries; in that case
            `packet_offsets` must give the starting sample index of each
            binary so per-binary sample indexing maps to the merged timeline.
        pairs: output of `pair_rpts`.
        samples_per_packet: typically 50 (ICD).
        dt: seconds per ADC sample (0.01 for ASIO).
        packet_offsets: optional list of `(packet_start_index, sample_offset)`
            per binary segment of `headers`. If None, treats all headers as
            one continuous stream starting at sample 0.

    Returns:
        list[{"t0", "t1", "label", "auto"}]

    Raises ValueError if any pair has no packets in its MUSE wall-clock
    window (per the agreed UX: refuse to load when an rpt doesn't match
    the binary).
    """
    if not pairs:
        return []
    if not headers:
        raise ValueError("no packet headers — empty binary?")

    muse_sec = np.array([h["muse_sec"] for h in headers], dtype=np.int64)
    asio_ms = np.array([h["asio_time_ms"] for h in headers], dtype=np.int64)
    first_ms = np.array([h["first_dp_ms"] for h in headers], dtype=np.int64)

    segments = []
    for i, pair in enumerate(pairs, start=1):
        s, e = pair["start"], pair["end"]
        in_window = (muse_sec >= s["muse_sec"]) & (muse_sec <= e["muse_sec"])
        cand = np.where(in_window)[0]
        if cand.size == 0:
            raise ValueError(
                f"No packets in MUSE wall-clock window for test {i} "
                f"({s['name']} → {e['name']}): "
                f"start TIME_SECONDS={s['muse_sec']}, end TIME_SECONDS={e['muse_sec']}. "
                "The .rpt files don't match this binary."
            )
        cand_asio = asio_ms[cand]
        start_pkt = int(cand[np.argmin(np.abs(cand_asio - s["asio_time_ms"]))])
        end_pkt = int(cand[np.argmin(np.abs(cand_asio - e["asio_time_ms"]))])

        # Sub-packet refinement: linearly interpolate first_dp_ms -> asio_time_ms
        # across the 50 samples, then pick the sample whose interpolated time
        # is closest to the .rpt's ASIO_TIME target.
        def _pkt_sample_times(idx):
            return np.linspace(first_ms[idx], asio_ms[idx],
                               samples_per_packet, endpoint=True)

        start_off = int(np.argmin(np.abs(_pkt_sample_times(start_pkt)
                                          - s["asio_time_ms"])))
        end_off = int(np.argmin(np.abs(_pkt_sample_times(end_pkt)
                                        - e["asio_time_ms"])))

        lo_sample = start_pkt * samples_per_packet + start_off
        hi_sample = end_pkt * samples_per_packet + end_off

        if packet_offsets is not None:
            # Remap (packet_idx, sample) through per-binary offsets so the
            # returned t0/t1 sit on the merged timeline.
            lo_sample = _remap_sample(start_pkt, start_off,
                                       samples_per_packet, packet_offsets)
            hi_sample = _remap_sample(end_pkt, end_off,
                                       samples_per_packet, packet_offsets)

        segments.append({
            "t0": lo_sample * dt,
            "t1": hi_sample * dt,
            "label": str(i),
            "auto": True,
        })

    return segments


def _remap_sample(pkt_idx, sample_off, samples_per_packet, packet_offsets):
    """Translate a (packet, sample-within-packet) pair from the merged
    headers list into a sample index on the merged sample timeline.

    `packet_offsets` is a list of (first_packet_idx, sample_offset) tuples,
    one per stitched binary, in order.
    """
    # Find the binary whose packet range contains pkt_idx.
    binary_first_pkt = 0
    sample_offset = 0
    for first_pkt, samp_off in packet_offsets:
        if pkt_idx >= first_pkt:
            binary_first_pkt = first_pkt
            sample_offset = samp_off
    local_pkt = pkt_idx - binary_first_pkt
    return sample_offset + local_pkt * samples_per_packet + sample_off
