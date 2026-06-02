"""ASIO EM analysis CLI package."""

# Force a non-interactive matplotlib backend so that the CLI never pops up GUI
# windows that block on a keypress. Must happen BEFORE any submodule imports
# pyplot.
import matplotlib as _matplotlib
_matplotlib.use("Agg", force=True)

__version__ = "0.1.0"
