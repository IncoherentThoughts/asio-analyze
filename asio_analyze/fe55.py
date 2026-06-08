"""Fe-55 analysis helpers.

The raw statistics + raw voltage emission live in `commands.cmd_fe55` and
reuse the existing helpers in `noise_analysis`. The Fe-55 *expectation values*
computation does not exist yet; `expectation_values()` is a stub so the CLI
and PDF assembly have a stable place to call into.
"""


def expectation_values(df):
    """Compute Fe-55 expectation values from a (6, N) voltage DataFrame.

    Not yet implemented. Returns None and prints a notice so callers can
    surface the placeholder state in their CSV / PDF output.
    """
    print("Fe-55 expectation values: not yet implemented")
    return None
