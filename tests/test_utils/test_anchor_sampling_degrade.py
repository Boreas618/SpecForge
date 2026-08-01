"""A micro-batch with no usable anchors must not raise.

The loss denominator is all-reduced across the world, so a rank that raises
inside anchor sampling exits before that collective and every peer blocks on it
until the NCCL timeout -- one degenerate micro-batch hangs the whole job instead
of failing it. Sampling must instead return a fully-masked batch that
contributes nothing and still reaches the collective.
"""

import unittest

import torch

from specforge.algorithms.common.dflash_family_model import (
    OnlineDFlashModel,
    OnlineDSparkModel,
)


class _BaseSampler:
    """Bind the unbound sampler to a stub carrying only the attributes it reads."""

    def __init__(self, block_size=7, num_anchors=512):
        self.block_size = block_size
        self.num_anchors = num_anchors


class _DSparkSampler(_BaseSampler):
    def _build_anchor_candidate_mask(self, seq_len, loss_mask):
        # Mirrors the real helper: position p is a candidate when p and p+1 are
        # both supervised.
        n = max(seq_len - 1, 0)
        return (loss_mask[:, :n] > 0.5) & (loss_mask[:, 1 : n + 1] > 0.5)


class TestAnchorSamplingDegrade(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.seq_len = 16

    def test_dflash_no_valid_anchor_returns_masked_batch(self):
        stub = _BaseSampler()
        loss_mask = torch.zeros(2, self.seq_len)  # nothing supervised
        anchors, keep = OnlineDFlashModel._sample_anchor_positions(
            stub, self.seq_len, loss_mask, self.device
        )
        self.assertEqual(anchors.shape[0], 2)
        self.assertEqual(keep.shape, anchors.shape)
        self.assertFalse(bool(keep.any()), "keep_mask must be entirely False")
        self.assertEqual(keep.dtype, torch.bool)
        self.assertEqual(anchors.dtype, torch.long)

    def test_dspark_no_adjacent_pair_returns_masked_batch(self):
        stub = _DSparkSampler()
        # Supervised tokens exist but are isolated, so no p with p and p+1 both
        # supervised -- the case that appears after non-finite sanitization.
        loss_mask = torch.zeros(2, self.seq_len)
        loss_mask[:, ::2] = 1.0
        anchors, keep = OnlineDSparkModel._sample_anchor_positions(
            stub, self.seq_len, loss_mask, self.device
        )
        self.assertFalse(bool(keep.any()), "keep_mask must be entirely False")
        self.assertEqual(anchors.shape, keep.shape)

    def test_dspark_normal_case_still_samples(self):
        stub = _DSparkSampler()
        loss_mask = torch.ones(2, self.seq_len)
        anchors, keep = OnlineDSparkModel._sample_anchor_positions(
            stub, self.seq_len, loss_mask, self.device
        )
        self.assertTrue(bool(keep.any()), "a fully supervised batch must yield anchors")


if __name__ == "__main__":
    unittest.main()
