"""Compatibility patches for importing LeRobot in the xhum-new environment.

The installed lerobot source contains dataclass definitions that are rejected by
Python 3.12's stricter ordering rules (non-default fields after default fields).
This module applies a minimal, targeted runtime patch to ``dataclasses.dataclass``
so that the affected class can be decorated without modifying lerobot source files.
"""

from __future__ import annotations

import dataclasses
import functools
import logging

logger = logging.getLogger(__name__)


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


def _patch_pi05_vision_tower_key_remap() -> None:
    """Remap PI05 vision-tower checkpoint keys when they carry a ``vision_model.`` infix.

    Some pi05_base checkpoints store SigLIP weights as
    ``...vision_tower.vision_model.*`` while the installed PI05Policy registers
    them as ``...vision_tower.*``; without remapping, the whole vision tower is
    silently left randomly initialized. Only remaps when this mismatch is
    actually detected; otherwise the state dict passes through unchanged.
    """
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    orig_fix = PI05Policy._fix_pytorch_state_dict_keys
    if getattr(orig_fix, "_galbot_vision_remap", False):
        return

    INFIX = "vision_tower.vision_model."

    def _fix_with_vision_remap(self, state_dict, model_config):
        fixed = orig_fix(self, state_dict, model_config)
        if not any(INFIX in key for key in fixed):
            return fixed
        model_keys = {name for name, _ in self.named_parameters()}
        model_keys.update(name for name, _ in self.named_buffers())
        expects_plain = any("vision_tower." in name and INFIX not in name for name in model_keys)
        if not expects_plain:
            return fixed
        remapped = {key.replace(INFIX, "vision_tower."): value for key, value in fixed.items()}
        logger.info(
            "PI05 vision tower key mismatch detected; remapped %d keys ('%s' -> 'vision_tower.')",
            sum(INFIX in key for key in fixed),
            INFIX,
        )
        return remapped

    _fix_with_vision_remap._galbot_vision_remap = True
    PI05Policy._fix_pytorch_state_dict_keys = _fix_with_vision_remap


def apply_lerobot_compat_patch() -> None:
    """Apply the dataclass compatibility patch before importing lerobot policies."""
    dataclasses.dataclass = _patched_dataclass
    _patch_pi05_vision_tower_key_remap()
