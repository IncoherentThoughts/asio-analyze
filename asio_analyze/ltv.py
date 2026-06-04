"""Light Tightness Verification analysis.

Compares two segments from the same dataset: an LPT (reference) segment
captured earlier and a DATA segment captured later. For each of the six
ASIO channels, reports the relative difference in the per-segment mean
voltage, ``(mean_LPT - mean_DATA) / mean_LPT``.
"""

import numpy as np


CHANNEL_NAMES = ['SXR1', 'SXR2', 'SXR3', 'SXR4', 'HXR', 'EUV']


def LTV(LPT, DATA):
    """Per-channel relative mean differences between an LPT and DATA segment.

    Both inputs are ``(N, 6)`` arrays with channels in
    ``SXR1, SXR2, SXR3, SXR4, HXR, EUV`` order; ``N`` may differ between
    the two segments.

    Returns a dict ``{channel: relative_difference_float}``.
    """
    means_LPT = np.empty(6)
    means_DATA = np.empty(6)
    for i in range(6):
        means_LPT[i] = np.mean(LPT[:, i])
        means_DATA[i] = np.mean(DATA[:, i])

    R = (means_LPT - means_DATA) / means_LPT
    return {name: float(R[i]) for i, name in enumerate(CHANNEL_NAMES)}


def evaluate_ltv(lpt_array, data_array):
    """Run :func:`LTV` and also return a per-channel summary table.

    ``lpt_array`` and ``data_array`` are both ``(N, 6)``. Returns
    ``(results, summary_rows)`` where ``results`` is the dict from
    :func:`LTV` and each ``summary_rows`` entry is
    ``[channel, mean_LPT, mean_DATA, relative_difference]``.
    """
    results = LTV(lpt_array, data_array)
    summary_rows = []
    for i, name in enumerate(CHANNEL_NAMES):
        mean_lpt = float(np.mean(lpt_array[:, i]))
        mean_data = float(np.mean(data_array[:, i]))
        summary_rows.append([name, mean_lpt, mean_data, results[name]])
    return results, summary_rows
