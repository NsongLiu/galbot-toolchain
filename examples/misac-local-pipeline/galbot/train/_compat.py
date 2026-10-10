"""Compatibility patches for importing LeRobot in the policy environment.

The installed lerobot source contains dataclass definitions that are rejected by
Python 3.12's stricter ordering rules (non-default fields after default fields).
This module applies a minimal, targeted runtime patch to ``dataclasses.dataclass``
so that the affected class can be decorated without modifying lerobot source files.
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np
import torch


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


def apply_lerobot_runtime_compat_patch() -> None:
    """Patch small LeRobot runtime incompatibilities for the local torch stack."""
    try:
        import lerobot.utils.random_utils as random_utils
    except Exception:
        return

    def _serialize_numpy_rng_state() -> dict[str, torch.Tensor]:
        np_state = np.random.get_state()
        assert np_state[0] == "MT19937"
        return {
            "np_rng_state_values": torch.tensor(np_state[1].astype(np.int64), dtype=torch.int64),
            "np_rng_state_index": torch.tensor([np_state[2]], dtype=torch.int64),
            "np_rng_has_gauss": torch.tensor([np_state[3]], dtype=torch.int64),
            "np_rng_cached_gaussian": torch.tensor([np_state[4]], dtype=torch.float32),
        }

    random_utils.serialize_numpy_rng_state = _serialize_numpy_rng_state
