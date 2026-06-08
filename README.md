# asio-analyze

CLI and browser GUI for analyzing ASIO Engineering Model test data. Wraps the
noise-analysis, detrending, segment-discovery, and PDF-generation routines
behind a single installable command. Commands are organized around the
physical tests run in the lab.

## Install

From this directory:

```bash
pip install -e .
```

This exposes two commands on your `PATH`:

- `asio-analyze` — terminal CLI / interactive wizard
- `asio-gui` — local browser GUI (stdlib HTTP server, opens your default browser)

## CLI usage

Run with no arguments to launch the interactive wizard:

```bash
asio-analyze
```

Or invoke a specific command directly. `<path>` is either a single ASIO CSV
file or a directory containing one or more ASIO CSV files.

| Command | What it does |
|---|---|
| `asio-analyze <path>` | **Default.** Per-trial CSV with current stats in fA (mean / std / RMS), trial duration, and raw voltages per channel. PDF only when `--pdf` is passed. |
| `asio-analyze background <path>` | Background test: detrended + EMI-filtered current stats, raw voltages, noise histograms. CSV + PDF. |
| `asio-analyze ltv <path> --lpt-start T0 --lpt-end T1 --data-start T2 --data-end T3` | Light Tightness Verification: compares an earlier LPT reference segment to a later DATA segment from the same file, reporting per-channel mean current in each window and their relative difference. CSV + PDF. |
| `asio-analyze fe55 <path>` | Fe-55 test: raw current stats (not detrended / not EMI-filtered), expectation values (stub), raw voltages. CSV + PDF. |
| `asio-analyze full <path>` | Full report: detrended + EMI-filtered stats (mean/std/RMS/skew/kurtosis), raw stats, duration, raw voltages, FFTs, EMI-corrected voltages, noise histograms, and the Fe-55 expectation-values stub. CSV + PDF. |

### Common options

- `--note "TEXT"` — attaches a free-text note to PDFs produced by the run.
- `--output-dir DIR` — overrides the default output location (`<data dir>/analysis`).
- `--pdf` *(default command only)* — also emit a PDF report; otherwise default produces CSV only.
- `--section-offset N` — start section numbering at `N+1` so the report can be appended after section `N` of a parent document. Useful when stitching multiple per-test reports into a single deliverable.
- `--lpt-start / --lpt-end / --data-start / --data-end` *(`ltv` only)* — segment windows in seconds. The LTV command no longer uses z-score thresholding; it compares the mean current in the LPT window against the DATA window directly.

Run `asio-analyze <command> --help` for the full option list per command.

## GUI usage

```bash
asio-gui
```

Opens a local web page that accepts raw ASIO captures rather than pre-decoded
CSVs. You drop in:

- one or more binary files named `YYYYMMDD.0xHEX` (consecutive dates are
  stitched), and
- the matching Galaxy GSE `.rpt` files (each test produces a
  `*_asio_start_test.rpt` / `*_asio_end_test.rpt` pair).

The GUI parses the binaries, pairs the `.rpt` files into `(t0, t1)` segments
via MUSE wall-clock matching, and lets you pick segments to feed into the same
analysis commands as the CLI.

## Output files

Per input CSV, named with the input's basename, all written under the
analysis directory:

- **default:** `<base>_stats.csv` (`channel, mean_fA, rms_fA, std_fA, duration_s`), `<base>_voltages.csv`, optional `PDF_<base>.pdf`.
- **background:** `<base>_background_stats.csv` (`channel, mean_fA, rms_fA, std_fA`), `<base>_voltages.csv`, `PDF_background_<base>.pdf`.
- **ltv:** `<base>_ltv_stats.csv` (`channel, mean_LPT_V, mean_DATA_V, relative_difference`), `<base>_voltages.csv`, `PDF_ltv_<base>.pdf`.
- **fe55:** `<base>_fe55_stats.csv`, `<base>_voltages.csv`, `PDF_fe55_<base>.pdf`.
- **full:** `<base>_full_raw_stats.csv`, `<base>_full_detrended_stats.csv`, `<base>_full_ltv_stats.csv`, `<base>_voltages.csv`, `PDF_full_<base>.pdf`.

Statistics are reported as currents in femtoamps (fA), converted from voltage
via the per-channel transimpedance: SXR1–4 = 112 MΩ, HXR = 9 MΩ,
EUV = 7.8 MΩ. The `<base>_voltages.csv` files contain the underlying raw
voltages (in volts) for traceability.

## Package layout

```
asio_analyze/
├── cli.py             # argparse + interactive wizard
├── gui.py             # local web GUI (stdlib http.server)
├── commands.py        # one function per subcommand
├── noise_analysis.py  # core analysis & plot functions
├── latex_report.py    # LaTeX/PDF report builders (one per command)
├── ltv.py             # Light Tightness Verification (LPT vs DATA window compare)
├── fe55.py            # Fe-55 expectation values (stub)
├── get_data.py        # ASIO binary packet parsing
├── rpt.py             # Galaxy GSE .rpt parsing and start/end pairing
└── segments.py        # build the segment program from .rpt pairs

scripts/
├── analysis_pipeline.ipynb   # exploratory end-to-end notebook
├── csv_generator.py          # synthetic test-data generator (validation only)
├── encoding/                 # binary/packet encoding experiments
├── ltv_experiment/           # LTV algorithm prototypes
└── segmentizing/             # .rpt-based segment-discovery prototypes

data/
├── binaryfile.txt   # drop raw ASIO binaries here (YYYYMMDD.0xHEX). The
│                    # included file is an empty placeholder — replace it
│                    # with the real captures before running the GUI.
├── RPTs/            # matching Galaxy GSE .rpt files (start/end pairs)
└── misc/            # everything else: pre-decoded CSVs, scratch files,
                     # ad-hoc exports
```

The `data/` tree is a scaffold, not a fixed input path — every CLI command
takes an explicit `<path>` and the GUI lets you browse to a location, so you
can keep captures anywhere. The convention here just keeps binaries and their
paired `.rpt` files together so the GUI's segment matcher has everything it
needs in one drop.
