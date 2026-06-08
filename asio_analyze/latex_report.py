"""LaTeX-based report builders for the asio-analyze CLI.

Reports are styled to match existing ASIO documentation (helvet sans-serif,
fancy header band with doc number, booktabs + longtable, figure[H] floats)
so they can be appended to existing test documents as an appendix.

Public surface:
    create_default_report(filename, stats_basic, duration_s, raw_voltages_img,
                          subtitle=None, note=None, section_offset=0)
    create_background_report(filename, stats_basic, raw_voltages_img,
                             cleaned_hist_img, subtitle=None, note=None,
                             section_offset=0)
    create_fe55_report(filename, stats_basic, raw_voltages_img,
                       subtitle=None, note=None, section_offset=0)
    create_ltv_report(filename, ltv_summary, lpt_window, data_window,
                      raw_voltages_img, subtitle=None, note=None,
                      section_offset=0)
    create_full_report(filename, stats_raw_full, stats_detrended_full,
                       duration_s, image_paths, subtitle=None, note=None,
                       section_offset=0)

`section_offset` (default 0) emits `\\setcounter{section}{N}` so the report's
section numbers continue from a parent document when appended.

Each `filename` must end in `.pdf`; a `.tex` file with the same stem is
written alongside it for transparency / debugging.
"""

import os
import re
import shutil
import subprocess
import tempfile
from datetime import date

from . import get_data


# ---------------------------------------------------------------------------
# Shared input prep
# ---------------------------------------------------------------------------

_ANALYSIS_CHANNEL_NAMES = ("SXR1", "SXR2", "SXR3", "SXR4", "HXR", "EUV")


def _is_analysis_format(file_path):
    """True when the CSV is already in the (channel-name, samples...) layout
    used by analysis. Detected by checking that the first column of the first
    row is one of the canonical channel names."""
    try:
        with open(file_path, "rb") as f:
            raw = f.read(128)
    except OSError:
        return False
    if not raw:
        return False
    try:
        first = raw.decode("utf-8").splitlines()[0]
    except (UnicodeDecodeError, IndexError):
        return False  # binary file: definitely not the analysis layout
    head = first.split(",", 1)[0].strip()
    return head in set(_ANALYSIS_CHANNEL_NAMES)


def _prepare_analysis_frame(file_path):
    """Parse an ASIO capture into an in-memory (6, N) voltage DataFrame.

    The frame has one row per channel in canonical order, with no
    channel-name column. If `file_path` is already in the channel-name-first
    CSV layout, it is read directly; otherwise the binary parser is used.
    """
    import pandas as pd

    if _is_analysis_format(file_path):
        df = pd.read_csv(file_path, header=None)
        df = df.drop(df.columns[0], axis=1).reset_index(drop=True)
        df.columns = range(df.shape[1])
        return df
    d = get_data.get_data_dict(file_path)
    return pd.DataFrame([d[name] for name in _ANALYSIS_CHANNEL_NAMES])


# ---------------------------------------------------------------------------
# pdflatex discovery
# ---------------------------------------------------------------------------

_CANDIDATE_PDFLATEX = [
    "/Library/TeX/texbin/pdflatex",
    "/usr/local/texlive/2025basic/bin/universal-darwin/pdflatex",
    "/usr/local/texlive/2025/bin/universal-darwin/pdflatex",
    "/usr/local/texlive/2024/bin/universal-darwin/pdflatex",
    "/usr/local/texlive/2023/bin/universal-darwin/pdflatex",
    "/usr/local/texlive/2022/bin/universal-darwin/pdflatex",
]


def _find_pdflatex():
    for path in _CANDIDATE_PDFLATEX:
        if os.path.exists(path):
            return path
    on_path = shutil.which("pdflatex")
    if on_path:
        return on_path
    raise RuntimeError(
        "pdflatex not found. Install a TeX distribution (TeX Live or MiKTeX) "
        "and ensure `pdflatex` is on PATH."
    )


# ---------------------------------------------------------------------------
# LaTeX preamble
# ---------------------------------------------------------------------------

PREAMBLE_TEMPLATE = r"""\documentclass[12pt]{article}
\usepackage[scaled]{helvet}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\usepackage[english]{babel}
\usepackage{graphicx}
\usepackage{ragged2e}
\usepackage{float}
\usepackage{setspace}
\usepackage[skip=12pt]{parskip}
\usepackage{enumitem}
\usepackage[font={small,it}]{caption}
\usepackage{hyperref}
\usepackage{url}
\usepackage{underscore}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\usepackage{xcolor,colortbl}
\usepackage{fancyhdr}

\renewcommand\familydefault{\sfdefault}
\captionsetup{justification=centering,singlelinecheck=false,format=hang}
\geometry{letterpaper, margin=1in}
\graphicspath{ {./} }
\pagenumbering{arabic}
\renewcommand\arraystretch{1.5}
\setlength\LTleft\fill
\setlength\LTright\fill

\newcommand{\PreserveBackslash}[1]{\let\temp=\\#1\let\\=\temp}
\newcolumntype{C}[1]{>{\PreserveBackslash\centering}p{#1}}
\newcolumntype{R}[1]{>{\PreserveBackslash\raggedleft}p{#1}}
\newcolumntype{L}[1]{>{\PreserveBackslash\raggedright}p{#1}}

\newcommand{\docAbr}{__DOC_ABR__}
\newcommand{\docNum}{__DOC_NUM__}
\newcommand{\dateEdit}{__DATE__}
\newcommand{\revNum}{0.0}

\newgeometry{top=1.25in,left=1in,right=1in,bottom=1in,headheight=0.8in,headsep=24pt}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\textcolor{gray}{\docAbr} \vspace{6pt}}
\fancyhead[C]{\textcolor{gray}{\docNum} \vspace{6pt}}
\fancyhead[R]{\textcolor{gray}{\dateEdit} \vspace{6pt}}
\fancyfoot[R]{Page: \hspace{6pt} \thepage}
\fancyfoot[L]{\textcolor{gray}{Revision~\revNum}}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

\captionsetup{belowskip=0pt}
\captionsetup{aboveskip=10pt}
"""


def _doc_number_from_csv(csv_basename):
    """Make a header doc-number tag like `ASIO-ANALYSIS-01-20250918-BG-LIGHTS-1`.

    Dashes (not underscores) are used because the value is interpolated into
    `\\textcolor{gray}{...}` in the page header where `_` would trigger a
    math-mode error.
    """
    stem = os.path.splitext(os.path.basename(csv_basename))[0]
    safe = re.sub(r"[^A-Za-z0-9-]+", "-", stem).strip("-").upper()
    return f"ASIO-ANALYSIS-{safe}"


def _preamble(csv_subtitle):
    doc_num = _doc_number_from_csv(csv_subtitle) if csv_subtitle else "ASIO-ANALYSIS"
    return (PREAMBLE_TEMPLATE
            .replace("__DOC_ABR__", "ASIO ANALYSIS")
            .replace("__DOC_NUM__", doc_num)
            .replace("__DATE__", date.today().strftime("%m/%d/%Y")))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_sci(x):
    """Format a float as `$m.mmmm \times 10^{n}$` (LaTeX math mode)."""
    if x == 0:
        return r"$0$"
    s = f"{x:.4e}"
    mant, exp = s.split("e")
    return f"${mant} \\times 10^{{{int(exp)}}}$"


def _tex_escape(text):
    """Minimal escape for free-form note text in body paragraphs."""
    if text is None:
        return ""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    out = text
    for a, b in replacements:
        out = out.replace(a, b)
    return out


def _stats_longtable(stats, columns, caption, label):
    """Render a stats longtable.

    `columns` is the header row including 'Channel' as the first column.
    `stats` rows are [name, val1, val2, ...] aligned to `columns[1:]`.
    All numeric values are rendered via `_fmt_sci`.
    """
    n_cols = len(columns)
    if n_cols < 2:
        raise ValueError("need at least Channel + 1 data column")
    # Column widths: first column 0.9in, remaining share equally up to ~5.4in
    data_w = round(5.4 / (n_cols - 1), 2)
    col_spec = "|C{0.9in}|" + "|".join([f"C{{{data_w}in}}"] * (n_cols - 1)) + "|"
    header = " & ".join(f"\\textbf{{{c}}}" for c in columns) + r" \\"

    body_rows = []
    for row in stats:
        name = row[0]
        cells = [f"\\textbf{{{name}}}"] + [_fmt_sci(v) for v in row[1:n_cols]]
        body_rows.append(" & ".join(cells) + r" \\")
    body = "\n\\hline\n".join(body_rows)

    return rf"""
\begin{{table}}[H]
\centering
\caption{{{caption}}} \label{{{label}}}
\begin{{tabular}}{{{col_spec}}}
\hline
{header}
\hline
{body}
\hline
\end{{tabular}}
\end{{table}}
"""


def _figure(img_basename, caption, label):
    return rf"""
\begin{{figure}}[H]
    \centering
    \includegraphics[width=6in]{{{img_basename}}}
    \caption[{caption}]{{{caption}}}
    \label{{{label}}}
\end{{figure}}
"""


def _duration_block(duration_s):
    return rf"""
\begin{{center}}
\begin{{tabular}}{{|C{{2in}}|C{{2in}}|}}
\hline
\textbf{{Quantity}} & \textbf{{Value}} \\
\hline
Total acquisition time & {duration_s:.3f}~s \\
\hline
\end{{tabular}}
\end{{center}}
"""


def _notes_block(note):
    if not note:
        return ""
    return rf"""
\subsection*{{Testing Notes}}
{_tex_escape(note)}
"""


def _section_offset_line(section_offset):
    if not section_offset:
        return ""
    return f"\\setcounter{{section}}{{{int(section_offset)}}}\n"


# ---------------------------------------------------------------------------
# pdflatex driver
# ---------------------------------------------------------------------------

def _compile_to_pdf(tex_source, out_pdf_path, image_paths):
    """Compile `tex_source` to `out_pdf_path`.

    Images named by basename in the .tex are copied into the build dir so
    `\\includegraphics{<basename>}` finds them. The generated `.tex` is also
    placed next to `out_pdf_path` for debugging.
    """
    pdflatex = _find_pdflatex()
    out_dir = os.path.dirname(os.path.abspath(out_pdf_path))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(out_pdf_path))[0]

    with tempfile.TemporaryDirectory(prefix="asio_latex_") as build_dir:
        tex_path = os.path.join(build_dir, f"{stem}.tex")
        with open(tex_path, "w") as f:
            f.write(tex_source)

        for img in image_paths:
            if img and os.path.exists(img):
                shutil.copy(img, os.path.join(build_dir, os.path.basename(img)))

        for _ in range(2):
            result = subprocess.run(
                [pdflatex, "-interaction=nonstopmode", "-halt-on-error",
                 f"{stem}.tex"],
                cwd=build_dir, capture_output=True, text=True,
            )
            if result.returncode != 0:
                log = os.path.join(build_dir, f"{stem}.log")
                tail = ""
                if os.path.exists(log):
                    with open(log) as f:
                        tail = f.read()[-2000:]
                raise RuntimeError(
                    f"pdflatex failed for {out_pdf_path}\n"
                    f"--- stdout tail ---\n{result.stdout[-1500:]}\n"
                    f"--- log tail ---\n{tail}"
                )

        built_pdf = os.path.join(build_dir, f"{stem}.pdf")
        shutil.move(built_pdf, out_pdf_path)
        # Drop the .tex next to the .pdf for transparency
        shutil.copy(tex_path, os.path.join(out_dir, f"{stem}.tex"))


# ---------------------------------------------------------------------------
# Body templates
# ---------------------------------------------------------------------------

STATS_COLS_BASIC = ["Channel", "Mean (fA)", "RMS (fA)", "Std Dev (fA)"]
STATS_COLS_FULL = ["Channel", "Mean (fA)", "RMS (fA)", "Std Dev (fA)",
                   "Skew", "Kurtosis"]


def _ltv_relative_diff_table(summary_rows, lpt_window, data_window, label):
    """Render the LPT-vs-DATA per-channel relative-difference table.

    `summary_rows` shape: [[channel, mean_lpt, mean_data, rel_diff], ...].
    """
    header_cells = [
        "Channel",
        r"$\overline{V}_{\mathrm{LPT}}$ (V)",
        r"$\overline{V}_{\mathrm{DATA}}$ (V)",
        r"$(\overline{V}_{\mathrm{LPT}} - \overline{V}_{\mathrm{DATA}}) / "
        r"\overline{V}_{\mathrm{LPT}}$",
    ]
    col_spec = "|C{0.9in}|C{1.2in}|C{1.2in}|C{1.6in}|"
    header = " & ".join(f"\\textbf{{{c}}}" for c in header_cells) + r" \\"

    body_rows = []
    for name, mean_lpt, mean_data, rel_diff in summary_rows:
        cells = [
            f"\\textbf{{{name}}}",
            f"{mean_lpt:.4e}",
            f"{mean_data:.4e}",
            f"{rel_diff:+.4e}",
        ]
        body_rows.append(" & ".join(cells) + r" \\")
    body = "\n\\hline\n".join(body_rows)
    lpt_t0, lpt_t1 = lpt_window
    data_t0, data_t1 = data_window
    caption = (
        f"Per-channel relative difference between the LPT reference "
        f"segment $[{lpt_t0:.2f},\\,{lpt_t1:.2f}]$~s and the DATA segment "
        f"$[{data_t0:.2f},\\,{data_t1:.2f}]$~s."
    )

    return rf"""
\begin{{table}}[H]
\centering
\caption{{{caption}}} \label{{{label}}}
\begin{{tabular}}{{{col_spec}}}
\hline
{header}
\hline
{body}
\hline
\end{{tabular}}
\end{{table}}
"""


def _intro_paragraph(csv_subtitle, duration_s=None):
    """First paragraph for every report. Uses `\path{...}` for safe filename wrap."""
    name = csv_subtitle or "the supplied dataset"
    duration_clause = (
        f" The trial spans {duration_s:.2f}~s of continuous data acquisition "
        f"across all six channels."
        if duration_s is not None else ""
    )
    return (
        rf"This report summarizes the analysis of dataset \path{{{name}}}."
        + duration_clause
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def create_default_report(filename, stats_basic, duration_s, raw_voltages_img,
                          subtitle=None, note=None, section_offset=0):
    cap_raw = r"Raw voltage vs.\ time (s) for all six ASIO channels."
    offset = _section_offset_line(section_offset)
    intro = _intro_paragraph(subtitle, duration_s)
    stats_tbl = _stats_longtable(stats_basic, STATS_COLS_BASIC,
                                 "Raw Current Statistics", "tab:default_stats")
    fig_raw = _figure(os.path.basename(raw_voltages_img), cap_raw, "fig:default_raw")
    dur = _duration_block(duration_s)
    notes = _notes_block(note)
    body = rf"""
{offset}
\section{{ASIO Default Analysis Report}}

{intro}
Statistics below are raw (no detrending, no EMI filtering); units are femtoamps.

\subsection{{Current Statistics}}
{stats_tbl}

\subsection{{Raw Voltage Time Series}}
{fig_raw}

\subsection{{Trial Duration}}
{dur}
{notes}
"""
    tex = _preamble(subtitle) + "\n\\begin{document}\n" + body + "\n\\end{document}\n"
    _compile_to_pdf(tex, filename, [raw_voltages_img])


def create_background_report(filename, stats_basic, raw_voltages_img,
                             cleaned_hist_img, subtitle=None, note=None,
                             section_offset=0):
    cap_raw = r"Raw voltage vs.\ time (s) for all six ASIO channels."
    cap_hist = (r"Noise histograms after Savitzky--Golay detrending and "
                r"40~Hz EMI removal. Red curve is a Gaussian fit using "
                r"\texttt{scipy.stats.norm}.")
    offset = _section_offset_line(section_offset)
    intro = _intro_paragraph(subtitle)
    stats_tbl = _stats_longtable(stats_basic, STATS_COLS_FULL,
                                 "Detrended + EMI-filtered Current Statistics",
                                 "tab:bg_stats")
    fig_raw = _figure(os.path.basename(raw_voltages_img), cap_raw, "fig:bg_raw")
    fig_hist = _figure(os.path.basename(cleaned_hist_img), cap_hist, "fig:bg_hist")
    notes = _notes_block(note)
    body = rf"""
{offset}
\section{{ASIO Background Analysis Report}}

{intro}
Statistics below are computed after Savitzky--Golay detrending and removal of
the 40~Hz room-EMI spike. Units are femtoamps.

\subsection{{Current Statistics}}
{stats_tbl}

\subsection{{Raw Voltage Time Series}}
{fig_raw}

\subsection{{Noise Histograms}}
{fig_hist}
{notes}
"""
    tex = _preamble(subtitle) + "\n\\begin{document}\n" + body + "\n\\end{document}\n"
    _compile_to_pdf(tex, filename, [raw_voltages_img, cleaned_hist_img])


def create_fe55_report(filename, stats_basic, raw_voltages_img,
                       subtitle=None, note=None, section_offset=0):
    cap_raw = r"Raw voltage vs.\ time (s) for all six ASIO channels."
    offset = _section_offset_line(section_offset)
    intro = _intro_paragraph(subtitle)
    stats_tbl = _stats_longtable(stats_basic, STATS_COLS_BASIC,
                                 "Raw Current Statistics", "tab:fe55_stats")
    fig_raw = _figure(os.path.basename(raw_voltages_img), cap_raw, "fig:fe55_raw")
    notes = _notes_block(note)
    body = rf"""
{offset}
\section{{ASIO Fe-55 Analysis Report}}

{intro}
Statistics below are raw (no detrending, no EMI filtering); units are femtoamps.

\subsection{{Current Statistics}}
{stats_tbl}

\subsection{{Expectation Values}}
\textit{{Fe-55 expectation values: not yet implemented.}}

\subsection{{Raw Voltage Time Series}}
{fig_raw}
{notes}
"""
    tex = _preamble(subtitle) + "\n\\begin{document}\n" + body + "\n\\end{document}\n"
    _compile_to_pdf(tex, filename, [raw_voltages_img])


def create_ltv_report(filename, ltv_summary, lpt_window, data_window,
                      raw_voltages_img, subtitle=None, note=None,
                      section_offset=0):
    """Light Tightness Verification report.

    `ltv_summary` is the per-channel summary rows from
    :func:`asio_analyze.ltv.evaluate_ltv`. `lpt_window` and `data_window`
    are ``(t0, t1)`` tuples in seconds for the reference and tested segments.
    """
    cap_raw = r"Raw voltage vs.\ time (s) for all six ASIO channels."
    offset = _section_offset_line(section_offset)
    intro = _intro_paragraph(subtitle)
    tbl = _ltv_relative_diff_table(ltv_summary, lpt_window, data_window,
                                   "tab:ltv_relative_diff")
    fig_raw = _figure(os.path.basename(raw_voltages_img), cap_raw, "fig:ltv_raw")
    notes = _notes_block(note)
    lpt_t0, lpt_t1 = lpt_window
    data_t0, data_t1 = data_window
    body = rf"""
{offset}
\section{{ASIO Light Tightness Verification Report}}

{intro}
A reference (LPT) segment spanning $[{lpt_t0:.2f},\,{lpt_t1:.2f}]$~s is
compared against a later DATA segment spanning $[{data_t0:.2f},\,{data_t1:.2f}]$~s
from the same dataset. For each channel the table below reports the mean
voltage of each segment and the relative difference,
$(\overline{{V}}_{{\mathrm{{LPT}}}} - \overline{{V}}_{{\mathrm{{DATA}}}}) /
\overline{{V}}_{{\mathrm{{LPT}}}}$.

\subsection{{Relative Differences}}
{tbl}

\subsection{{Raw Voltage Time Series}}
{fig_raw}
{notes}
"""
    tex = _preamble(subtitle) + "\n\\begin{document}\n" + body + "\n\\end{document}\n"
    _compile_to_pdf(tex, filename, [raw_voltages_img])


def create_full_report(filename, stats_raw_full, stats_detrended_full,
                       duration_s, image_paths, subtitle=None, note=None,
                       section_offset=0):
    """`image_paths` order: [raw_voltages, fft, cleaned_voltages, cleaned_hist]."""
    img_raw, img_fft, img_cleaned, img_hist = image_paths
    cap_raw = r"Raw voltage vs.\ time (s) for all six ASIO channels."
    cap_fft = "FFTs of raw voltages (mean centered)."
    cap_cleaned = (r"Voltage vs.\ time (s) after Savitzky--Golay detrending "
                   r"and 40~Hz EMI removal.")
    cap_hist = (r"Noise histograms after detrending and 40~Hz EMI removal. "
                r"Red curve is a Gaussian fit.")
    offset = _section_offset_line(section_offset)
    intro = _intro_paragraph(subtitle, duration_s)
    stats_det = _stats_longtable(stats_detrended_full, STATS_COLS_FULL,
                                 "Detrended + EMI-filtered Current Statistics",
                                 "tab:full_det_stats")
    stats_raw = _stats_longtable(stats_raw_full, STATS_COLS_FULL,
                                 "Raw Current Statistics", "tab:full_raw_stats")
    dur = _duration_block(duration_s)
    fig_raw = _figure(os.path.basename(img_raw), cap_raw, "fig:full_raw")
    fig_fft = _figure(os.path.basename(img_fft), cap_fft, "fig:full_fft")
    fig_cleaned = _figure(os.path.basename(img_cleaned), cap_cleaned, "fig:full_cleaned")
    fig_hist = _figure(os.path.basename(img_hist), cap_hist, "fig:full_hist")
    notes = _notes_block(note)
    body = rf"""
{offset}
\section{{ASIO Full Analysis Report}}

{intro}
The remainder of this section reports both raw and detrended + EMI-filtered
statistics. Units are femtoamps for mean/RMS/std; skew and kurtosis are
dimensionless.

\subsection{{Detrended + EMI-filtered Statistics}}
{stats_det}

\subsection{{Raw Statistics}}
{stats_raw}

\subsection{{Trial Duration}}
{dur}

\subsection{{Raw Voltage Time Series}}
{fig_raw}

\subsection{{FFTs of Raw Voltages}}
{fig_fft}

\subsection{{Detrended + EMI-filtered Time Series}}
{fig_cleaned}

\subsection{{Noise Histograms}}
{fig_hist}

\subsection{{Fe-55 Expectation Values}}
\textit{{Fe-55 expectation values: not yet implemented.}}
{notes}
"""
    tex = _preamble(subtitle) + "\n\\begin{document}\n" + body + "\n\\end{document}\n"
    _compile_to_pdf(tex, filename, image_paths)
