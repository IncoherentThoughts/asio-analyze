"""Light Tightness Verification analysis.

Per-channel z-score anomaly check on the EMI-cleaned + detrended signal.
A channel passes when no sample exceeds `sensitivity` standard deviations
from the channel mean; otherwise it fails.

The z-score core algorithm matches :func:`asio_analyze.ltv_real.LTV`. This
module wraps that algorithm with the I/O adapter the rest of the pipeline
expects (transpose channel-rows into samples, prepend a time column),
collects per-channel anomaly summary stats, and writes CSV + PDF outputs.
"""

import os

import numpy as np

from . import ltv_real
from . import noise_analysis


CHANNEL_NAMES = ['SXR1', 'SXR2', 'SXR3', 'SXR4', 'HXR', 'EUV']
SAMPLE_DT_S = 0.01  # ASIO sampling period (100 Hz)


def _df_to_ltv_array(df):
    """Turn a (6 channels x N samples) DataFrame into LTV's (N, 7) layout.

    Output col 0 is time in seconds; cols 1..6 are SXR1..EUV in that order.
    """
    samples = df.to_numpy().T  # (N, 6)
    n = samples.shape[0]
    time = (np.arange(n) * SAMPLE_DT_S).reshape(-1, 1)
    return np.hstack([time, samples])


def evaluate_ltv(data, sensitivity):
    """Return (passfail_dict, summary_rows) for a (N, 7) LTV input array.

    `passfail_dict` is whatever :func:`ltv_real.LTV` returns (canonical pass/fail).
    `summary_rows` is `[[channel, verdict, count, t_first_s, t_last_s, max_abs_z], ...]`
    in the same channel order as ``CHANNEL_NAMES``.
    """
    passfail = ltv_real.LTV(data, sensitivity)

    times = data[:, 0]
    summary = []
    for i, name in enumerate(CHANNEL_NAMES):
        col = data[:, i + 1]
        mean = float(np.mean(col))
        std = float(np.std(col))
        if std == 0:
            zscores = np.zeros_like(col)
        else:
            zscores = (col - mean) / std
        mask = np.abs(zscores) > sensitivity
        flagged_times = times[mask]
        flagged_z = zscores[mask]
        count = int(flagged_times.size)
        t_first = float(flagged_times.min()) if count else float('nan')
        t_last = float(flagged_times.max()) if count else float('nan')
        max_abs_z = float(np.max(np.abs(flagged_z))) if count else 0.0
        summary.append([name, passfail[name], count, t_first, t_last, max_abs_z])
    return passfail, summary


def evaluate_ltv_from_cleaned(cleaned_df, sensitivity):
    """Run LTV on a cleaned (6xN) DataFrame from :func:`noise_analysis.remove_room_emi`."""
    return evaluate_ltv(_df_to_ltv_array(cleaned_df), sensitivity)


def evaluate_ltv_from_csv(analysis_csv, sensitivity):
    """Run LTV directly from an analysis-CSV path (handles EMI removal internally)."""
    cleaned_df, _ = noise_analysis.remove_room_emi(analysis_csv)
    return evaluate_ltv_from_cleaned(cleaned_df, sensitivity)


# Backwards-compatible no-op kept so older imports of `run_ltv` don't break.
def run_ltv(directory, output_dir=None, note=None, sensitivity=4.0):
    """Deprecated thin wrapper. The real orchestration lives in
    :func:`asio_analyze.commands.cmd_ltv` which produces CSV + PDF artifacts.
    Calling this only runs the analysis in-memory and returns the result so
    legacy callers don't crash.
    """
    from .latex_report import _prepare_analysis_csv
    analysis_csv = _prepare_analysis_csv(directory) if os.path.isfile(directory) else None
    if analysis_csv is None:
        return None
    return evaluate_ltv_from_csv(analysis_csv, sensitivity)
