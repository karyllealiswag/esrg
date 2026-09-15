"""
__init__.py — ESRG package entry point.

Purpose : Expose the public API (Config, run, Result, Stage) and the version.
Function : Re-exports from config.py and pipeline.py so callers use `from esrg
          import Config, run` without reaching into submodules.
Notes   : Enhanced Seeded Region Growing for MRI brain tumor segmentation.
"""
from .config import Config
from .pipeline import run, Result, Stage

__all__ = ["Config", "run", "Result", "Stage"]
__version__ = "1.0.0"
