import unittest

import torch

from specforge.core.dconv import create_dconv_sdpa_mask


def _reference_mask(anchors, keep, causal, context_len, stream_len):
    batch_size, num_blocks = anchors.shape
    query_len = num_blocks * stream_len
    kv_len = context_len + query_len
    mask = torch.zeros(batch_size, 1, query_len, kv_len, dtype=torch.bool)
    for batch_idx in range(batch_size):
        for query_idx in range(query_len):
            block_idx = query_idx // stream_len
            if not keep[batch_idx, block_idx]:
                continue
            for kv_idx in range(kv_len):
                if kv_idx < context_len:
                    visible = kv_idx < anchors[batch_idx, block_idx]
                else:
                    kv_block_idx = (kv_idx - context_len) // stream_len
                    visible = kv_block_idx == block_idx
                    if visible and causal[batch_idx, block_idx]:
                        visible = (
                            kv_idx - context_len
                        ) % stream_len <= query_idx % stream_len
                mask[batch_idx, 0, query_idx, kv_idx] = visible
    return mask


class DConvMaskTest(unittest.TestCase):
    def test_mixed_passes_and_invalid_blocks(self):
        anchors = torch.tensor([[4, 9, 0], [3, 7, 11]])
        keep = torch.tensor([[True, True, False], [True, False, True]])
        causal = torch.tensor([[False, True, False], [True, False, False]])
        actual = create_dconv_sdpa_mask(
            anchor_positions=anchors,
            block_keep_mask=keep,
            causal_block_mask=causal,
            S=12,
            stream_len=5,
            device=torch.device("cpu"),
        )
        expected = _reference_mask(anchors, keep, causal, 12, 5)
        self.assertTrue(torch.equal(actual, expected))

    def test_causal_stream_cannot_read_self_or_future_input_for_prediction(self):
        mask = create_dconv_sdpa_mask(
            anchor_positions=torch.tensor([[6]]),
            block_keep_mask=torch.tensor([[True]]),
            causal_block_mask=torch.tensor([[True]]),
            S=10,
            stream_len=7,
            device=torch.device("cpu"),
        )
        stream_mask = mask[0, 0, :, 10:]
        self.assertTrue(torch.equal(stream_mask, torch.ones(7, 7).tril().bool()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
