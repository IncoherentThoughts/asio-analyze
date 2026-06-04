"""One function per asio-analyze subcommand.

Each `cmd_*` takes already-validated arguments and writes CSV (always) +
PDF (conditionally) outputs into the resolved analysis directory.

Commands map to physical tests rather than analysis types:
    default     - lightweight per-trial summary (voltage stats + duration + raw voltages)
    background  - detrended + EMI-filtered analysis
    ltv         - Light Tightness Verification (stub)
    fe55        - Fe-55 test (raw stats + raw voltages + expectation-values stub)
    full        - superset of everything above
"""

import csv
import glob
import os
import shutil

import numpy as np
import pandas as pd

from . import noise_analysis
from . import fe55 as fe55_module
from . import ltv as ltv_module
from . import latex_report as report_module
from .latex_report import _prepare_analysis_csv


LTV_STATS_COLS = ['channel', 'mean_LPT_V', 'mean_DATA_V', 'relative_difference']
LTV_SAMPLE_DT_S = 0.01  # ASIO sampling period; matches get_data.SAMPLE_DT


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _list_csvs(directory):
    return sorted(glob.glob(os.path.join(directory, '*.csv')))


def _resolve_inputs(path):
    """Accept either a directory of CSVs or a single CSV file."""
    if os.path.isfile(path):
        return [path], os.path.dirname(os.path.abspath(path))
    if os.path.isdir(path):
        return _list_csvs(path), path
    raise ValueError(f"'{path}' is not a file or directory")


def _cleanup_misc(directory):
    misc_dir = os.path.join(directory, 'misc')
    if os.path.exists(misc_dir):
        shutil.rmtree(misc_dir)


def _setup_run(directory, output_dir):
    csv_files, anchor_dir = _resolve_inputs(directory)
    analysis_dir = output_dir or os.path.join(anchor_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    print(f"Found {len(csv_files)} CSV file(s) under {directory}")
    return csv_files, anchor_dir, analysis_dir


def _write_note(note, analysis_dir):
    if not note:
        return
    notes_path = os.path.join(analysis_dir, 'run_note.txt')
    with open(notes_path, 'w') as f:
        f.write(note + "\n")
    print(f"Saved note to {notes_path}")


def _voltages_df(analysis_csv):
    df = pd.read_csv(analysis_csv, header=None)
    df = df.drop(df.columns[0], axis=1)
    return df


def _duration_seconds(n_samples):
    """Replicate `np.arange(N) * 0.01` -> last sample time in seconds."""
    if n_samples <= 0:
        return 0.0
    return float((n_samples - 1) * 0.01)


def _write_stats_csv(path, stats, columns):
    """`stats` rows are [channel, val1, val2, ...] aligned with `columns`."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in stats:
            writer.writerow([row[0]] + [f"{v:.10e}" for v in row[1:]])
    print(f"Wrote {path}")


def _write_voltages_csv(path, analysis_csv):
    df = pd.read_csv(analysis_csv, header=None)
    n_cols = df.shape[1] - 1
    df.columns = ['channel'] + [f"t{i}" for i in range(n_cols)]
    df.to_csv(path, index=False)
    print(f"Wrote {path}")


def _select_stats(stats_full, indices):
    """Project the 6-tuple [name, mean, rms, std, skew, kurt] down to subset."""
    return [[row[0]] + [row[i] for i in indices] for row in stats_full]


# Transimpedance values (ohms) per channel - matches noise_analysis.Std_and_RMS_current_comparison
TRANSIMPEDANCE = {
    'SXR1': 112e6, 'SXR2': 112e6, 'SXR3': 112e6, 'SXR4': 112e6,
    'HXR': 9e6, 'EUV': 7.8e6,
}


def _to_current_fA(stats):
    """Convert voltage stats to current stats in femtoamps.

    Input rows are [name, mean_V, rms_V, std_V[, skew, kurt]].
    Mean / RMS / StdDev scale by (1e15 / transimpedance). Skew and kurtosis
    are dimensionless under linear scaling so they pass through unchanged.
    """
    out = []
    for row in stats:
        name = row[0]
        z = TRANSIMPEDANCE[name]
        scale = 1e15 / z
        new_row = [name]
        # mean (1), rms (2), std (3) - scale to fA
        for i in (1, 2, 3):
            if i < len(row):
                new_row.append(row[i] * scale)
        # skew (4), kurtosis (5) - unchanged
        for i in (4, 5):
            if i < len(row):
                new_row.append(row[i])
        out.append(new_row)
    return out


# ---------------------------------------------------------------------------
# default
# ---------------------------------------------------------------------------

STATS_COLS_BASIC_CSV = ['channel', 'mean_fA', 'rms_fA', 'std_fA', 'duration_s']
STATS_COLS_FULL_CSV = ['channel', 'mean_fA', 'rms_fA', 'std_fA', 'skew', 'kurtosis']
STATS_COLS_BG_FE55_CSV = ['channel', 'mean_fA', 'rms_fA', 'std_fA']


def cmd_default(directory, output_dir=None, note=None, emit_pdf=False,
                section_offset=0, **_):
    csv_files, anchor_dir, analysis_dir = _setup_run(directory, output_dir)

    for csv_file in csv_files:
        base = os.path.splitext(os.path.basename(csv_file))[0]
        print(f"\n[default] processing {csv_file} ...")
        try:
            analysis_csv = _prepare_analysis_csv(csv_file)
            stats_v = noise_analysis.channel_voltage_stats(analysis_csv)
            stats_basic = _to_current_fA(_select_stats(stats_v, [1, 2, 3]))

            n_samples = _voltages_df(analysis_csv).shape[1]
            duration_s = _duration_seconds(n_samples)

            # stats CSV with duration as trailing column (repeated per channel)
            stats_rows = [row + [duration_s] for row in stats_basic]
            stats_path = os.path.join(analysis_dir, f"{base}_stats.csv")
            _write_stats_csv(stats_path, stats_rows, STATS_COLS_BASIC_CSV)

            volt_path = os.path.join(analysis_dir, f"{base}_voltages.csv")
            _write_voltages_csv(volt_path, analysis_csv)

            if emit_pdf:
                img = noise_analysis.plot_signals_voltages(analysis_csv)
                pdf_path = os.path.join(analysis_dir, f"PDF_{base}.pdf")
                report_module.create_default_report(
                    pdf_path, stats_basic, duration_s, img,
                    subtitle=os.path.basename(csv_file), note=note,
                    section_offset=section_offset,
                )
                print(f"Wrote {pdf_path}")
        except Exception as e:
            print(f"[default] failed on {csv_file}: {e}")

    _write_note(note, analysis_dir)
    _cleanup_misc(anchor_dir)


# ---------------------------------------------------------------------------
# background
# ---------------------------------------------------------------------------

def cmd_background(directory, output_dir=None, note=None, emit_pdf=True,
                   section_offset=0, **_):
    csv_files, anchor_dir, analysis_dir = _setup_run(directory, output_dir)

    for csv_file in csv_files:
        base = os.path.splitext(os.path.basename(csv_file))[0]
        print(f"\n[background] processing {csv_file} ...")
        try:
            analysis_csv = _prepare_analysis_csv(csv_file)
            cleaned_df, _rms = noise_analysis.remove_room_emi(analysis_csv)
            stats_v = noise_analysis.channel_voltage_stats_detrended(cleaned_df)
            stats_basic = _to_current_fA(_select_stats(stats_v, [1, 2, 3, 4, 5]))

            stats_path = os.path.join(analysis_dir, f"{base}_background_stats.csv")
            _write_stats_csv(stats_path, stats_basic, STATS_COLS_FULL_CSV)

            volt_path = os.path.join(analysis_dir, f"{base}_voltages.csv")
            _write_voltages_csv(volt_path, analysis_csv)

            raw_img = noise_analysis.plot_signals_voltages(analysis_csv)
            hist_img = noise_analysis.plot_cleaned_histogram(cleaned_df, analysis_csv)

            if emit_pdf:
                pdf_path = os.path.join(analysis_dir, f"PDF_background_{base}.pdf")
                report_module.create_background_report(
                    pdf_path, stats_basic, raw_img, hist_img,
                    subtitle=os.path.basename(csv_file), note=note,
                    section_offset=section_offset,
                )
                print(f"Wrote {pdf_path}")
        except Exception as e:
            print(f"[background] failed on {csv_file}: {e}")

    _write_note(note, analysis_dir)
    _cleanup_misc(anchor_dir)


# ---------------------------------------------------------------------------
# ltv
# ---------------------------------------------------------------------------

def _write_ltv_stats_csv(path, summary_rows):
    """Write LTV LPT-vs-DATA per-channel relative-difference CSV."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(LTV_STATS_COLS)
        for name, mean_lpt, mean_data, rel_diff in summary_rows:
            writer.writerow([
                name,
                f"{mean_lpt:.10e}",
                f"{mean_data:.10e}",
                f"{rel_diff:.10e}",
            ])
    print(f"Wrote {path}")


def _slice_segment(df, t0, t1, dt=LTV_SAMPLE_DT_S):
    """Slice the (6 channels x N samples) voltage frame into an (N_seg, 6)
    array covering the closed-open time window ``[t0, t1)`` in seconds."""
    i0 = max(0, int(round(t0 / dt)))
    i1 = min(df.shape[1], int(round(t1 / dt)))
    if i1 <= i0:
        raise ValueError(
            f"empty segment slice: t0={t0:.3f}s, t1={t1:.3f}s -> [{i0}, {i1})"
        )
    return df.iloc[:, i0:i1].to_numpy().T  # (N_seg, 6)


def cmd_ltv(directory, lpt_window, data_window, output_dir=None, note=None,
            emit_pdf=True, section_offset=0, **_):
    """Light Tightness Verification: compare an LPT reference segment to a
    later DATA segment from the same file.

    ``lpt_window`` and ``data_window`` are ``(t0, t1)`` tuples in seconds.
    """
    csv_files, anchor_dir, analysis_dir = _setup_run(directory, output_dir)

    lpt_t0, lpt_t1 = float(lpt_window[0]), float(lpt_window[1])
    data_t0, data_t1 = float(data_window[0]), float(data_window[1])
    if data_t0 < lpt_t1:
        print(f"[ltv] warning: DATA segment starts at {data_t0:.2f}s, "
              f"before LPT ends at {lpt_t1:.2f}s")

    for csv_file in csv_files:
        base = os.path.splitext(os.path.basename(csv_file))[0]
        print(f"\n[ltv] processing {csv_file} "
              f"(LPT={lpt_t0:.2f}-{lpt_t1:.2f}s, "
              f"DATA={data_t0:.2f}-{data_t1:.2f}s) ...")
        try:
            analysis_csv = _prepare_analysis_csv(csv_file)
            voltages = _voltages_df(analysis_csv)
            lpt_arr = _slice_segment(voltages, lpt_t0, lpt_t1)
            data_arr = _slice_segment(voltages, data_t0, data_t1)
            results, summary = ltv_module.evaluate_ltv(lpt_arr, data_arr)
            print(f"  relative differences: {results}")

            stats_path = os.path.join(analysis_dir, f"{base}_ltv_stats.csv")
            _write_ltv_stats_csv(stats_path, summary)

            volt_path = os.path.join(analysis_dir, f"{base}_voltages.csv")
            _write_voltages_csv(volt_path, analysis_csv)

            raw_img = noise_analysis.plot_signals_voltages(analysis_csv)
            if emit_pdf:
                pdf_path = os.path.join(analysis_dir, f"PDF_ltv_{base}.pdf")
                report_module.create_ltv_report(
                    pdf_path, summary,
                    (lpt_t0, lpt_t1), (data_t0, data_t1),
                    raw_img,
                    subtitle=os.path.basename(csv_file), note=note,
                    section_offset=section_offset,
                )
                print(f"Wrote {pdf_path}")
        except Exception as e:
            print(f"[ltv] failed on {csv_file}: {e}")

    _write_note(note, analysis_dir)
    _cleanup_misc(anchor_dir)


# ---------------------------------------------------------------------------
# fe55
# ---------------------------------------------------------------------------

def cmd_fe55(directory, output_dir=None, note=None, emit_pdf=True,
             section_offset=0, **_):
    csv_files, anchor_dir, analysis_dir = _setup_run(directory, output_dir)

    for csv_file in csv_files:
        base = os.path.splitext(os.path.basename(csv_file))[0]
        print(f"\n[fe55] processing {csv_file} ...")
        try:
            analysis_csv = _prepare_analysis_csv(csv_file)
            stats_v = noise_analysis.channel_voltage_stats(analysis_csv)
            stats_basic = _to_current_fA(_select_stats(stats_v, [1, 2, 3]))

            stats_path = os.path.join(analysis_dir, f"{base}_fe55_stats.csv")
            _write_stats_csv(stats_path, stats_basic, STATS_COLS_BG_FE55_CSV)

            # Stub: expectation values not yet implemented
            fe55_module.expectation_values(analysis_csv)

            volt_path = os.path.join(analysis_dir, f"{base}_voltages.csv")
            _write_voltages_csv(volt_path, analysis_csv)

            raw_img = noise_analysis.plot_signals_voltages(analysis_csv)
            if emit_pdf:
                pdf_path = os.path.join(analysis_dir, f"PDF_fe55_{base}.pdf")
                report_module.create_fe55_report(
                    pdf_path, stats_basic, raw_img,
                    subtitle=os.path.basename(csv_file), note=note,
                    section_offset=section_offset,
                )
                print(f"Wrote {pdf_path}")
        except Exception as e:
            print(f"[fe55] failed on {csv_file}: {e}")

    _write_note(note, analysis_dir)
    _cleanup_misc(anchor_dir)


# ---------------------------------------------------------------------------
# full
# ---------------------------------------------------------------------------

def cmd_full(directory, output_dir=None, note=None, emit_pdf=True,
             section_offset=0, **_):
    csv_files, anchor_dir, analysis_dir = _setup_run(directory, output_dir)

    for csv_file in csv_files:
        base = os.path.splitext(os.path.basename(csv_file))[0]
        print(f"\n[full] processing {csv_file} ...")
        try:
            analysis_csv = _prepare_analysis_csv(csv_file)

            stats_raw_v = noise_analysis.channel_voltage_stats(analysis_csv)
            cleaned_df, _rms = noise_analysis.remove_room_emi(analysis_csv)
            stats_detrended_v = noise_analysis.channel_voltage_stats_detrended(cleaned_df)
            stats_raw = _to_current_fA(stats_raw_v)
            stats_detrended = _to_current_fA(stats_detrended_v)

            n_samples = _voltages_df(analysis_csv).shape[1]
            duration_s = _duration_seconds(n_samples)

            stats_path_raw = os.path.join(analysis_dir, f"{base}_full_raw_stats.csv")
            _write_stats_csv(stats_path_raw, stats_raw, STATS_COLS_FULL_CSV)

            stats_path_det = os.path.join(analysis_dir, f"{base}_full_detrended_stats.csv")
            _write_stats_csv(stats_path_det, stats_detrended, STATS_COLS_FULL_CSV)

            volt_path = os.path.join(analysis_dir, f"{base}_voltages.csv")
            _write_voltages_csv(volt_path, analysis_csv)

            img_raw = noise_analysis.plot_signals_voltages(analysis_csv)
            img_fft = noise_analysis.plot_fft(analysis_csv)
            img_cleaned = noise_analysis.plot_cleaned_signals_time_series(cleaned_df, analysis_csv)
            img_hist = noise_analysis.plot_cleaned_histogram(cleaned_df, analysis_csv)

            fe55_module.expectation_values(analysis_csv)

            if emit_pdf:
                pdf_path = os.path.join(analysis_dir, f"PDF_full_{base}.pdf")
                report_module.create_full_report(
                    pdf_path, stats_raw, stats_detrended, duration_s,
                    [img_raw, img_fft, img_cleaned, img_hist],
                    subtitle=os.path.basename(csv_file), note=note,
                    section_offset=section_offset,
                )
                print(f"Wrote {pdf_path}")
        except Exception as e:
            print(f"[full] failed on {csv_file}: {e}")

    _write_note(note, analysis_dir)
    _cleanup_misc(anchor_dir)
