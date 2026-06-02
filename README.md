# asio-analyze

CLI for analyzing ASIO Engineering Model test data. Wraps the noise-analysis,
detrending, and PDF-generation scripts behind a single installable command.
Commands are organized around the physical tests conducted in the lab.

## Install

From this directory:

```bash
pip install -e .
```

This exposes the `asio-analyze` command on your `PATH`.

## Usage

Run with no arguments to launch the interactive wizard:

```bash
asio-analyze
```

Or invoke a specific command directly. `<path>` is either a single ASIO CSV
file or a directory containing one or more ASIO CSV files.

| Command | What it does |
|---|---|
| `asio-analyze <path>` | **Default.** Per-trial CSV containing current stats in fA (mean / std / RMS), trial duration in seconds, and the raw voltages per channel. PDF only when `--pdf` is passed. |
| `asio-analyze background <path>` | Background test: detrended + EMI-filtered current stats (front and center), raw voltages, noise histograms (detrended + EMI filtered). CSV + PDF. |
| `asio-analyze ltv <path>` | Light Tightness Verification: per-channel pass/fail by z-score on the Savitzky--Golay detrended + 40 Hz EMI-filtered signal. A channel fails if any sample exceeds `--sensitivity` (default 4.0) standard deviations from its own mean. CSV + PDF. |
| `asio-analyze fe55 <path>` | Fe-55 test: raw current stats (not detrended / not EMI-filtered) front and center, expectation values (stub, not yet implemented), raw voltages. CSV + PDF. |
| `asio-analyze full <path>` | Full report: detrended + EMI-filtered current stats (mean/std/RMS/skew/kurtosis), raw current stats, duration, raw voltages, FFTs of raw voltages, detrended + EMI-corrected voltages, noise histograms, LTV pass/fail (at `--sensitivity`, default 4.0), Fe-55 expectation-values stub. CSV + PDF. |

### Common options

- `--note "TEXT"` — attaches a free-text note to any PDFs produced by the run.
- `--output-dir DIR` — overrides the default output location (`<data dir>/analysis`).
- `--pdf` *(default command only)* — also emit a PDF report; otherwise default produces CSV only.
- `--sensitivity FLOAT` *(`ltv` and `full` only)* — z-score threshold for LTV anomaly detection. Default `4.0`.

Run `asio-analyze <command> --help` for the full option list per command.

## Output files

Per input CSV, named with the input's basename, all written to the
analysis directory:

- **default:** `<base>_stats.csv` (channel, mean_fA, rms_fA, std_fA, duration_s), `<base>_voltages.csv`, optional `PDF_<base>.pdf`.
- **background:** `<base>_background_stats.csv`, `<base>_voltages.csv`, `PDF_background_<base>.pdf`.
- **ltv:** `<base>_ltv_stats.csv` (channel, pass_fail, anomaly_count, t_first_s, t_last_s, max_abs_z), `<base>_voltages.csv`, `PDF_ltv_<base>.pdf`.
- **fe55:** `<base>_fe55_stats.csv`, `<base>_voltages.csv`, `PDF_fe55_<base>.pdf`.
- **full:** `<base>_full_raw_stats.csv`, `<base>_full_detrended_stats.csv`, `<base>_full_ltv_stats.csv`, `<base>_voltages.csv`, `PDF_full_<base>.pdf`.

Statistics are reported as currents in femtoamps (fA), converted from voltage
via the per-channel transimpedance: SXR1-4 = 112 MOhm, HXR = 9 MOhm,
EUV = 7.8 MOhm. The `<base>_voltages.csv` files contain the underlying raw
voltages (in volts) for traceability.

## Package layout

```
asio_analyze/
├── cli.py             # argparse + interactive wizard
├── commands.py        # one function per subcommand
├── noise_analysis.py  # core analysis & plot functions
├── pdf.py             # PDF builders (one per command)
├── ltv.py             # Light Tightness Verification (stub)
├── fe55.py            # Fe-55 expectation values (stub)
├── get_data.py        # ASIO packet parsing
└── csv_generator.py   # synthetic test-data generator (validation only)
```
