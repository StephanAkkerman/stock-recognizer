"""The package version, in one place.

Deliberately a module of its own containing nothing but a literal. Both
`pyproject.toml` (via ``[tool.setuptools.dynamic]``) and the package read from
here, so a release is one edit rather than two values that can drift apart.

A bare literal in a leaf module is read statically by setuptools at build time,
so resolving the version never imports the package or needs its dependencies.
"""

__version__ = "0.2.0"
