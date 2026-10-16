"""Choose where TensorFlow runs (only the stage-1 CNN uses TensorFlow).

TensorFlow fixes its devices the first time it runs an op, so this must be
called before any model is built. StageOneClassifier does that itself, so
users normally only set ``ClassifierConfig.device``.
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)
_applied: Optional[tuple] = None


def configure_tensorflow(device: str = "cpu", intra_op_threads: Optional[int] = None,
                         inter_op_threads: Optional[int] = None) -> str:
    """Apply the device choice; returns a short description of what TensorFlow will use.

    device="cpu"   hide all GPUs (CUDA or Apple Metal) from TensorFlow
    device="gpu"   require at least one GPU; raise if there is none
    device="auto"  leave TensorFlow's default (GPU if one is visible)
    """
    global _applied
    if device not in ("cpu", "gpu", "auto"):
        raise ValueError(f"device must be 'cpu', 'gpu' or 'auto', not {device!r}")
    key = (device, intra_op_threads, inter_op_threads)

    import tensorflow as tf

    if _applied != key:
        gpus = tf.config.list_physical_devices("GPU")
        if device == "gpu" and not gpus:
            raise RuntimeError("device='gpu' but TensorFlow sees no GPU; use 'cpu' or 'auto'")
        try:
            if device == "cpu" and gpus:
                tf.config.set_visible_devices([], "GPU")
            if intra_op_threads:
                tf.config.threading.set_intra_op_parallelism_threads(intra_op_threads)
            if inter_op_threads:
                tf.config.threading.set_inter_op_parallelism_threads(inter_op_threads)
        except RuntimeError as exc:  # TensorFlow already initialised in this process
            log.warning("could not apply device=%r (%s); TensorFlow keeps its current devices",
                        device, exc)
        _applied = key

    visible_gpus = tf.config.get_visible_devices("GPU")
    desc = f"TensorFlow on {'GPU x%d' % len(visible_gpus) if visible_gpus else 'CPU'} (device={device!r})"
    log.info(desc)
    return desc
