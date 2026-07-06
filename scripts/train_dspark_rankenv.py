"""Per-rank env wrapper for train_dspark.py.

Forces a 45-minute timeout on EVERY torch.distributed process group created
after this point (sglang's tp/moe sub-groups otherwise get the 10-minute torch
default; a rank stalled in kernel compilation/autotune can legitimately exceed
that and must not bring the whole run down).
"""

import os
import runpy
from datetime import timedelta

_TIMEOUT = timedelta(minutes=45)

import torch.distributed as dist  # noqa: E402
from torch.distributed import distributed_c10d as _c10d  # noqa: E402

_orig_new_group = _c10d.new_group


def _patched_new_group(*args, **kwargs):
    kwargs["timeout"] = _TIMEOUT
    return _orig_new_group(*args, **kwargs)


_c10d.new_group = _patched_new_group
dist.new_group = _patched_new_group

_orig_init_pg = _c10d.init_process_group


def _patched_init_pg(*args, **kwargs):
    kwargs["timeout"] = _TIMEOUT
    return _orig_init_pg(*args, **kwargs)


_c10d.init_process_group = _patched_init_pg
dist.init_process_group = _patched_init_pg

runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_dspark.py"),
    run_name="__main__",
)
