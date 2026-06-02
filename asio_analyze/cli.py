"""Command-line entry point for asio-analyze.

Usage:
    asio-analyze <path> [--pdf] [--note TEXT] [--output-dir DIR]   # default
    asio-analyze background <path> [--note TEXT] [--output-dir DIR]
    asio-analyze ltv        <path> [--note TEXT] [--output-dir DIR]
    asio-analyze fe55       <path> [--note TEXT] [--output-dir DIR]
    asio-analyze full       <path> [--note TEXT] [--output-dir DIR]
    asio-analyze                                                   # wizard
"""

import argparse
import os
import sys

from . import __version__
from . import commands


MENU = [
    ("default",    "Default lightweight summary: voltage stats + duration + raw voltages (CSV only by default)."),
    ("background", "Background test: detrended + EMI-filtered stats, raw voltages, noise histograms."),
    ("ltv",        "Light Tightness Verification: per-channel pass/fail by z-score on the EMI-cleaned signal."),
    ("fe55",       "Fe-55 test: raw stats, expectation values (stub), raw voltages."),
    ("full",       "Full report: every analysis section, including LTV pass/fail."),
]


LTV_DEFAULT_SENSITIVITY = 4.0


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _dispatch(name, params):
    if name == "default":
        return commands.cmd_default(**params)
    if name == "background":
        return commands.cmd_background(**params)
    if name == "ltv":
        return commands.cmd_ltv(**params)
    if name == "fe55":
        return commands.cmd_fe55(**params)
    if name == "full":
        return commands.cmd_full(**params)
    raise ValueError(f"Unknown command: {name}")


# ---------------------------------------------------------------------------
# argparse setup
# ---------------------------------------------------------------------------

DEFAULT_USAGE_EPILOG = """\
default mode (no subcommand):
  asio-analyze <path> [--pdf] [--note TEXT] [--output-dir DIR]

  <path>            Path to a CSV file OR a directory containing ASIO CSV files
  --pdf             Also emit a PDF report (default: CSV only)
  --note TEXT       Free-text note embedded in generated PDFs
  --output-dir DIR  Where to write output (default: <directory>/analysis)

Run `asio-analyze` with no arguments to launch the interactive wizard.
"""


def _build_parser():
    p = argparse.ArgumentParser(
        prog="asio-analyze",
        description="ASIO test data analyzer. "
                    "Run without arguments for an interactive menu.",
        epilog=DEFAULT_USAGE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command")

    def add_dir_subparser(name, help_text, with_pdf_flag=False, with_sensitivity=False):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("directory", help="Path to a CSV file OR a directory containing ASIO CSV files")
        sp.add_argument("--note", default=None,
                        help="Free-text note embedded in generated PDFs")
        sp.add_argument("--output-dir", default=None,
                        help="Where to write output (default: <directory>/analysis)")
        sp.add_argument("--section-offset", type=int, default=0,
                        help="Start section numbering at N+1 so the report can be "
                             "appended after section N of a parent document "
                             "(default: 0)")
        if with_pdf_flag:
            sp.add_argument("--pdf", action="store_true",
                            help="Also emit a PDF report (default: CSV only)")
        if with_sensitivity:
            sp.add_argument("--sensitivity", type=float,
                            default=LTV_DEFAULT_SENSITIVITY,
                            help=f"LTV z-score threshold; samples with |z| above "
                                 f"this count as anomalies "
                                 f"(default: {LTV_DEFAULT_SENSITIVITY})")
        return sp

    add_dir_subparser("background", "Background test analysis")
    add_dir_subparser("ltv",        "Light Tightness Verification",
                      with_sensitivity=True)
    add_dir_subparser("fe55",       "Fe-55 test analysis")
    add_dir_subparser("full",       "Full report (all sections)",
                      with_sensitivity=True)

    return p


def _args_to_params(args):
    params = {
        "directory": args.directory,
        "output_dir": args.output_dir,
        "note": args.note,
        "section_offset": getattr(args, "section_offset", 0),
    }
    if args.command == "default":
        params["emit_pdf"] = getattr(args, "pdf", False)
    if args.command in ("ltv", "full"):
        params["sensitivity"] = getattr(args, "sensitivity",
                                        LTV_DEFAULT_SENSITIVITY)
    return params


# ---------------------------------------------------------------------------
# Interactive wizard (no-args mode)
# ---------------------------------------------------------------------------

def _prompt(label, default=None):
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{label}{suffix}: ").strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
        val = val[1:-1]
    if not val and default is not None:
        return default
    return val


def _prompt_yes_no(label, default=False):
    suffix = " [y/N]" if not default else " [Y/n]"
    while True:
        raw = input(f"{label}{suffix}: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please answer y or n.")


def _wizard():
    print("\n=== asio-analyze ===\n")
    print("Choose an analysis option:")
    for i, (name, desc) in enumerate(MENU, start=1):
        print(f"  {i}) {name:<10} - {desc}")
    print("  0) quit")
    print()

    while True:
        raw = input(f"Selection (0-{len(MENU)}): ").strip()
        if raw == "0":
            print("Cancelled.")
            return 0
        try:
            choice = int(raw)
            if 1 <= choice <= len(MENU):
                break
        except ValueError:
            pass
        print("  Please enter a number from the menu.")
    name = MENU[choice - 1][0]
    print(f"\nSelected: {name}\n")

    params = _wizard_dir_command(name)
    print("\nRunning ...\n")
    _dispatch(name, params)
    return 0


def _wizard_dir_command(name):
    while True:
        directory = _prompt("Data path (a CSV file, or a directory containing CSVs)")
        if directory and (os.path.isdir(directory) or os.path.isfile(directory)):
            directory = os.path.abspath(directory)
            print(f"  -> resolved to: {directory}")
            break
        print(f"  '{directory}' is not a valid file or directory.")
    note = _prompt("Optional note to embed in PDFs (blank = none)", default="")
    note = note or None
    output_dir = _prompt("Output directory (blank = <data dir>/analysis)", default="")
    if output_dir:
        output_dir = os.path.abspath(output_dir)
        print(f"  -> output dir resolved to: {output_dir}")
    else:
        output_dir = None
        default_out = os.path.join(
            directory if os.path.isdir(directory) else os.path.dirname(directory),
            'analysis',
        )
        print(f"  -> using default output dir: {default_out}")

    params = {"directory": directory, "note": note, "output_dir": output_dir}
    if name == "default":
        params["emit_pdf"] = _prompt_yes_no("Also emit a PDF?", default=False)
    if name in ("ltv", "full"):
        while True:
            raw = _prompt("LTV sensitivity (z-score threshold)",
                          default=str(LTV_DEFAULT_SENSITIVITY))
            try:
                params["sensitivity"] = float(raw)
                break
            except ValueError:
                print(f"  '{raw}' is not a valid number.")
    return params


# ---------------------------------------------------------------------------
# Default-mode parsing (no subcommand but a path is given)
# ---------------------------------------------------------------------------

def _parse_default_args(argv):
    """Parse `asio-analyze <path> [--pdf] [--note ...] [--output-dir ...]`.

    Returns a params dict or None if argv doesn't look like a default invocation.
    """
    parser = argparse.ArgumentParser(prog="asio-analyze", add_help=True)
    parser.add_argument("directory")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--note", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--section-offset", type=int, default=0)
    args = parser.parse_args(argv)
    return {
        "directory": args.directory,
        "output_dir": args.output_dir,
        "note": args.note,
        "emit_pdf": args.pdf,
        "section_offset": args.section_offset,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_KNOWN_SUBCOMMANDS = {"background", "ltv", "fe55", "full"}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        return _wizard()

    # If the first token is a known subcommand, use the subparser machinery.
    if argv[0] in _KNOWN_SUBCOMMANDS:
        parser = _build_parser()
        args = parser.parse_args(argv)
        params = _args_to_params(args)
        _dispatch(args.command, params)
        return 0

    # Help/version still go through the main parser.
    if argv[0] in ("-h", "--help", "--version"):
        parser = _build_parser()
        parser.parse_args(argv)
        return 0

    # Otherwise treat as the default command: first positional is <path>.
    params = _parse_default_args(argv)
    _dispatch("default", params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
