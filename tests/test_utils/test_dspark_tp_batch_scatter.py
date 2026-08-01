"""TP-batch scatter: each TP rank trains a distinct 1/tp_size batch slice.

With a co-located TP-replicated target every rank in the TP group receives the
same batch, so without slicing each rank runs the draft on the full node batch
-- tp_size redundant copies of the same compute. Slicing is bit-identical for
DSpark because its loss pools the denominator with a world all-reduce.
"""

import unittest
from unittest import mock

import torch
import torch.nn as nn

from specforge.training.strategies.base import DSparkTrainStrategy


def _batch(bsz=8, S=6, H=4):
    return {
        "input_ids": torch.arange(bsz * S).reshape(bsz, S),
        "hidden_states": torch.randn(bsz, S, H),
        "loss_mask": torch.ones(bsz, S),
        "target_last_hidden_states": torch.randn(bsz, S, H),
    }


def _strategy(**kw):
    return DSparkTrainStrategy(nn.Linear(1, 1), **kw)


class TestTpBatchScatter(unittest.TestCase):
    def test_no_distributed_passthrough(self):
        t = _batch()
        out = _strategy()._tp_slice(t)
        for k in t:
            self.assertTrue(torch.equal(out[k], t[k]))

    def test_flag_off_passthrough(self):
        t = _batch()
        out = _strategy(tp_batch_scatter=False)._tp_slice(t)
        for k in t:
            self.assertTrue(torch.equal(out[k], t[k]))

    def _sliced(self, t, tp_size, tp_rank):
        group = object()
        with mock.patch("torch.distributed.is_available", return_value=True), \
             mock.patch("torch.distributed.is_initialized", return_value=True), \
             mock.patch("torch.distributed.get_world_size", return_value=tp_size), \
             mock.patch("torch.distributed.get_rank", return_value=tp_rank), \
             mock.patch("specforge.distributed.get_tp_group", return_value=group):
            return _strategy()._tp_slice(t)

    def test_rank_slices_are_disjoint_and_cover_batch(self):
        t = _batch(bsz=8)
        out0 = self._sliced(t, tp_size=2, tp_rank=0)
        out1 = self._sliced(t, tp_size=2, tp_rank=1)
        for k in t:
            self.assertEqual(out0[k].shape[0], 4)
            self.assertEqual(out1[k].shape[0], 4)
            self.assertTrue(torch.equal(out0[k], t[k][:4]))
            self.assertTrue(torch.equal(out1[k], t[k][4:]))
            self.assertTrue(
                torch.equal(torch.cat([out0[k], out1[k]]), t[k]),
                f"{k}: rank slices must reassemble the full batch",
            )

    def test_all_tensors_sliced_consistently(self):
        # Rows must stay aligned across the four tensors after slicing.
        t = _batch(bsz=4)
        out = self._sliced(t, tp_size=2, tp_rank=1)
        row = t["input_ids"][2]
        self.assertTrue(torch.equal(out["input_ids"][0], row))
        self.assertTrue(torch.equal(out["hidden_states"][0], t["hidden_states"][2]))

    def test_indivisible_batch_passthrough(self):
        t = _batch(bsz=7)
        out = self._sliced(t, tp_size=2, tp_rank=0)
        for k in t:
            self.assertTrue(torch.equal(out[k], t[k]))


if __name__ == "__main__":
    unittest.main()
