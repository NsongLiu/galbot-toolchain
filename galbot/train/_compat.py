"""Compatibility patches for importing LeRobot in the xhum-new environment.

The installed lerobot source contains dataclass definitions that are rejected by
Python 3.12's stricter ordering rules (non-default fields after default fields).
This module applies a minimal, targeted runtime patch to ``dataclasses.dataclass``
so that the affected class can be decorated without modifying lerobot source files.
"""

from __future__ import annotations

import dataclasses
import functools


_ORIG_DATACLASS = dataclasses.dataclass
_ORIG_FIELD = dataclasses.field


def _patched_field(*, default=dataclasses.MISSING, default_factory=dataclasses.MISSING, init=None, **kwargs):
    """Give init=False fields a default value so Python 3.12 accepts the ordering."""
    if init is False and default is dataclasses.MISSING and default_factory is dataclasses.MISSING:
        default = None
    return _ORIG_FIELD(default=default, default_factory=default_factory, init=init, **kwargs)


@functools.wraps(_ORIG_DATACLASS)
def _patched_dataclass(cls=None, /, **kwargs):
    def wrap(cls):
        is_target = cls.__name__ == "GR00TN15Config"
        if is_target:
            dataclasses.field = _patched_field
        try:
            return _ORIG_DATACLASS(cls, **kwargs)
        finally:
            if is_target:
                dataclasses.field = _ORIG_FIELD

    if cls is None:
        return wrap
    return wrap(cls)


def apply_lerobot_compat_patch() -> None:
    """Apply the dataclass compatibility patch before importing lerobot policies."""
    dataclasses.dataclass = _patched_dataclass
