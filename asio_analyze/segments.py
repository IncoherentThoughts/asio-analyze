"""Build the segment program for one (possibly stitched) ASIO capture.

Segments are now derived from `.rpt` start/end pairs rather than HXR10/HXR11
relay edges. See `asio_analyze.rpt` for the algorithm; this module is a thin
adapter that the GUI calls with the parsed packet headers and a list of
`.rpt` paths.
"""

from __future__ import annotations

from . import rpt as _rpt


def build_segment_program(headers, dt, samples_per_packet, rpt_paths,
                          packet_offsets=None):
    """Pair the .rpt files and convert each pair into a (t0, t1) segment.

    Args:
        headers: per-packet header dicts (one per packet, in merged order).
        dt: seconds per ADC sample (typically 0.01).
        samples_per_packet: ICD-defined (50).
        rpt_paths: list of .rpt file paths supplied by the user.
        packet_offsets: optional list of (first_packet_idx, sample_offset)
            tuples, one per binary, to translate packet/sample indices to the
            stitched sample timeline. None ⇒ single-binary case.

    Returns:
        {"segments": [...], "errors": [{"path", "reason"}, ...]}

    Raises ValueError if any pair's MUSE wall-clock window doesn't match any
    packet in `headers` (the .rpt files don't belong to this binary).
    """
    if not rpt_paths:
        return {"segments": [], "errors": []}

    pairs, errors = _rpt.pair_rpts(list(rpt_paths))
    segments = _rpt.segments_from_rpts(
        headers, pairs, samples_per_packet, dt,
        packet_offsets=packet_offsets,
    )
    return {"segments": segments, "errors": errors}
