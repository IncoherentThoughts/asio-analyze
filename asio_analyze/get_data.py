"""ASIO binary parser.

Ported from ``flipsample.ipynb`` (the canonical, ICD-correct implementation).

ICD §5.1: 950-byte little-endian packet.

  Offset  Size  Field
  ------  ----  -----------------------------------------------------
  0       4     Start ID "ASIO"
  4       1     Relay Configuration (bit layout LSB->MSB):
                  EUV5, EUV6, EUV7, HXR8, HXR9, HXR10, HXR11, Unused
  5       10    Temperature Data (5 sensors x 2 bytes, 16-bit LE)
  15      1     Bad Command Count
  16      1     Padded Zero
  17      2     Voltage Data (16-bit LE)
  19      2     Current Data (16-bit LE)
  21      2     Packet Count (16-bit LE)
  23      1     Command Count
  24      6     MUSE Time (4 bytes seconds + 2 bytes sub-seconds, LE)
  30      2     Padded Zeros
  32      4     ASIO Time ms (32-bit LE)
  36      4     First Data Point ASIO Time ms (32-bit LE)
  40      4     Padded Zeros
  44      900   ADC Data (50 segments x 18 bytes = 50 segments x 6 channels x 3 bytes)
  944     4     End ID "STOP"
  948     2     CRC

Each channel sample is a 24-bit little-endian unsigned integer; ``ADC_to_V``
maps it to a voltage in [0, 5).
"""

import csv

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


PACKET_SIZE = 950
HEADER_SIZE = 44
ADC_SEGMENT_BYTES = 18           # 6 channels * 3 bytes
N_SEGMENTS = 50                  # ADC segments per packet
ADC_DATA_SIZE = N_SEGMENTS * ADC_SEGMENT_BYTES   # 900 bytes
STOP_OFFSET = HEADER_SIZE + ADC_DATA_SIZE        # 944
SAMPLE_DT = 0.01                 # seconds per ADC sample (10 ms)

CHANNEL_NAMES = ("SXR1", "SXR2", "SXR3", "SXR4", "HXR", "EUV")


# ---------------------------------------------------------------------------
# Byte helpers
# ---------------------------------------------------------------------------

def ADC_to_V(val):
    """Convert a 24-bit ADC integer to a voltage.

    Two's-complement upper half maps to negative offset around 2.5 V midpoint.
    Works on Python ints or numpy arrays (vectorized via np.where).
    """
    arr = np.asarray(val).astype(np.float64)
    high = arr > 0x7FFFFF
    pos = arr / 0x800000 * 2.5 + 2.5
    neg = (arr - (0xFFFFFF + 1)) / 0x800000 * 2.5 + 2.5
    return np.where(high, neg, pos)


def _decode16le(buf):
    return int(buf[0]) | (int(buf[1]) << 8)


def _decode32le(buf):
    return (
        int(buf[0])
        | (int(buf[1]) << 8)
        | (int(buf[2]) << 16)
        | (int(buf[3]) << 24)
    )


def _find_bytes(target, data):
    """Indices where the byte string ``target`` begins in ``data``."""
    target_bytes = np.frombuffer(target.encode(), dtype=np.uint8)
    if len(data) < len(target_bytes):
        return np.array([], dtype=int)
    windows = sliding_window_view(data, len(target_bytes))
    matches = (windows == target_bytes).all(axis=1)
    return np.where(matches)[0]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_raw_bytes(path):
    """Read the file as a uint8 array.

    Supports two on-disk formats:
      - Raw binary file (preferred; what the GSE writes today). Read via
        ``np.fromfile``.
      - Legacy CSV-of-bytes (a single row of comma-separated integers, one
        per byte). Detected by reading the first kilobyte and checking for
        ASCII digits/commas.
    """
    with open(path, "rb") as f:
        head = f.read(64)
    # Legacy CSV-of-bytes format: ASCII digits/commas only; never contains the
    # raw "ASIO" framing string.
    allowed = set(b"0123456789,-\r\n\t .")
    looks_like_csv = bool(head) and all(b in allowed for b in head)
    if looks_like_csv:
        with open(path, "r") as f:
            row = next(csv.reader(f))
        return np.array(row, dtype=int).astype(np.uint8)
    return np.fromfile(path, dtype=np.uint8)


def _collect_packets(raw):
    """Return a list of 950-byte uint8 packets validated by ASIO/STOP framing."""
    packets = []
    rejected = 0
    asio_indices = _find_bytes("ASIO", raw)
    for start in asio_indices:
        end = start + PACKET_SIZE
        if end > len(raw):
            rejected += 1
            continue
        stop_check = raw[start + STOP_OFFSET : start + STOP_OFFSET + 4].tobytes()
        if stop_check != b"STOP":
            rejected += 1
            continue
        packets.append(raw[start:end])
    if rejected:
        print(f"Skipped {rejected} malformed packet(s); kept {len(packets)}")
    return packets


# ---------------------------------------------------------------------------
# Per-packet parsing
# ---------------------------------------------------------------------------

def _parse_header(packet):
    relay = int(packet[4])
    temps = [_decode16le(packet[5 + i * 2 : 7 + i * 2]) for i in range(5)]
    return {
        "relay": relay,
        "temps_raw": temps,
        "bad_cmd_count": int(packet[15]),
        "voltage_raw": _decode16le(packet[17:19]),
        "current_raw": _decode16le(packet[19:21]),
        "packet_count": _decode16le(packet[21:23]),
        "cmd_count": int(packet[23]),
        "muse_sec": _decode32le(packet[24:28]),
        "muse_subsec": _decode16le(packet[28:30]),
        "asio_time_ms": _decode32le(packet[32:36]),
        "first_dp_ms": _decode32le(packet[36:40]),
    }


def _decode_adc(packets):
    """Decode all ADC samples for every channel.

    Returns a dict mapping channel name -> np.ndarray of voltages (float),
    length = n_packets * N_SEGMENTS.
    """
    n = len(packets)
    if n == 0:
        return {name: np.zeros(0, dtype=float) for name in CHANNEL_NAMES}

    block = np.stack([
        np.frombuffer(p[HEADER_SIZE : HEADER_SIZE + ADC_DATA_SIZE].tobytes(),
                      dtype=np.uint8)
        for p in packets
    ])
    block = block.reshape(n, N_SEGMENTS, 6, 3)
    adc = (
        block[..., 0].astype(np.uint32)
        | (block[..., 1].astype(np.uint32) << 8)
        | (block[..., 2].astype(np.uint32) << 16)
    )
    # adc shape: (n_packets, N_SEGMENTS, 6) -> per-channel flat voltage arrays
    out = {}
    for ch_idx, name in enumerate(CHANNEL_NAMES):
        out[name] = ADC_to_V(adc[..., ch_idx]).ravel().astype(float)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_data_dict(file):
    """Parse an ASIO capture (binary or legacy CSV) into a structured dict.

    Returned keys:
      SXR1, SXR2, SXR3, SXR4, HXR, EUV : np.ndarray(float)  voltages, length
                                                            n_packets * 50
      headers                 : list[dict]  per-packet housekeeping
                                (MUSE/ASIO times needed for .rpt segment mapping)
      n_packets               : int
      samples_per_packet      : 50
      dt                      : 0.01    (seconds per ADC sample)
    """
    raw = _load_raw_bytes(file)
    packets = _collect_packets(raw)
    if not packets:
        raise ValueError(f"No valid ASIO packets found in {file!r}")

    channels = _decode_adc(packets)
    headers = [_parse_header(p) for p in packets]

    out = dict(channels)
    out.update({
        "headers": headers,
        "n_packets": len(packets),
        "samples_per_packet": N_SEGMENTS,
        "dt": SAMPLE_DT,
    })
    return out


