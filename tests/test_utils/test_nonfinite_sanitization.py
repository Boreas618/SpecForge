"""Non-finite target features must be dropped from supervision, not propagated.

Target capture can emit non-finite hidden states for individual rows. Left
alone, one bad row NaNs the loss, and under data parallelism the gradient
all-reduce propagates that rank's NaN everywhere. Sanitization zeroes both the
loss mask and the feature for affected tokens so healthy tokens keep training.
"""

import unittest

import torch

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel


class _Stub:
    """Carries only the attribute _sanitize_nonfinite_inputs reads."""

    def __init__(self, sanitize=True):
        self.sanitize_nonfinite = sanitize

    _finite_report = staticmethod(OnlineDFlashModel._finite_report)
    _sanitize_nonfinite_inputs = OnlineDFlashModel._sanitize_nonfinite_inputs


class TestNonfiniteSanitization(unittest.TestCase):
    def setUp(self):
        self.B, self.S, self.H = 2, 8, 4

    def _clean_inputs(self):
        hidden = torch.randn(self.B, self.S, self.H)
        loss_mask = torch.ones(self.B, self.S)
        return hidden, loss_mask

    def test_finite_inputs_pass_through_unchanged(self):
        hidden, loss_mask = self._clean_inputs()
        h2, m2, t2 = _Stub()._sanitize_nonfinite_inputs(hidden, loss_mask)
        self.assertTrue(torch.equal(h2, hidden))
        self.assertTrue(torch.equal(m2, loss_mask))
        self.assertIsNone(t2)

    def test_nan_token_is_unsupervised_and_zeroed(self):
        hidden, loss_mask = self._clean_inputs()
        hidden[0, 3, 1] = float("nan")
        hidden[1, 5, 0] = float("inf")
        h2, m2, _ = _Stub()._sanitize_nonfinite_inputs(hidden, loss_mask)
        # Affected tokens: masked out, and their non-finite elements zeroed
        # (finite elements of a bad token may remain -- the token is
        # unsupervised, and only inf/NaN can leak through attention).
        self.assertEqual(float(m2[0, 3]), 0.0)
        self.assertEqual(float(m2[1, 5]), 0.0)
        self.assertEqual(float(h2[0, 3, 1]), 0.0)
        self.assertEqual(float(h2[1, 5, 0]), 0.0)
        self.assertTrue(torch.isfinite(h2).all())
        untouched = torch.ones_like(m2)
        untouched[0, 3] = 0.0
        untouched[1, 5] = 0.0
        self.assertTrue(torch.equal(m2, untouched))

    def test_bad_target_final_hidden_also_masks_token(self):
        hidden, loss_mask = self._clean_inputs()
        target_final = torch.randn(self.B, self.S, self.H)
        target_final[1, 2, 3] = float("nan")
        h2, m2, t2 = _Stub()._sanitize_nonfinite_inputs(
            hidden, loss_mask, target_final
        )
        self.assertEqual(float(m2[1, 2]), 0.0)
        self.assertTrue(torch.isfinite(t2).all())
        # Context hidden was finite there and must be preserved.
        self.assertTrue(torch.equal(h2[1, 2], hidden[1, 2]))

    def test_disabled_flag_passes_nan_through(self):
        hidden, loss_mask = self._clean_inputs()
        hidden[0, 0, 0] = float("nan")
        h2, m2, _ = _Stub(sanitize=False)._sanitize_nonfinite_inputs(
            hidden, loss_mask
        )
        self.assertTrue(bool(torch.isnan(h2).any()))
        self.assertTrue(torch.equal(m2, loss_mask))

    def test_all_bad_batch_yields_empty_supervision(self):
        # Composes with anchor-sampling degrade: a fully poisoned micro-batch
        # must end as all-masked, which sampling then tolerates.
        hidden = torch.full((self.B, self.S, self.H), float("nan"))
        loss_mask = torch.ones(self.B, self.S)
        h2, m2, _ = _Stub()._sanitize_nonfinite_inputs(hidden, loss_mask)
        self.assertEqual(float(m2.sum()), 0.0)
        self.assertTrue(torch.equal(h2, torch.zeros_like(h2)))


if __name__ == "__main__":
    unittest.main()
