"""Shim for editable installs on older pip versions that don't yet support
PEP 660 (pyproject-only editable installs). All real metadata lives in
pyproject.toml.
"""

from setuptools import setup

setup()
