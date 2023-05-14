"""Exports `simba_ml` directory as python package directory."""

from simba_ml import _version

__version__ = _version.get_versions()["version"]  # type: ignore[no-untyped-call]