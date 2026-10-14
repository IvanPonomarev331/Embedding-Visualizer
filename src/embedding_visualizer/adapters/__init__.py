"""Adapters that import external data into the canonical dataset format."""

from .npy import convert_npy

__all__ = ["convert_npy"]
