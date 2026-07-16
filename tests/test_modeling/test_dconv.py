import unittest

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.core.dconv import OnlineDConvModel, create_dconv_sdpa_mask
from specforge.modeling.auto import AutoEagle3DraftModel
from specforge.modeling.draft.dconv import DConvDraftModel, ShortConv
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.registry import available_drafts


def _tiny_config(projector_type="dconv"):
    config = Qwen3Config(
        vocab_size=257,
        hidden_size=64,
        intermediate_size=160,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
    )
    config.num_target_layers = 8
    config.block_size = 6
    config._attn_implementation = "sdpa"
    config.dflash_config = {
        "mask_token_id": 256,
        "projector_type": projector_type,
        "conv_width": 4,
        "window_len": 4,
    }
    return config


class DConvModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_zero_init_is_dflash_and_shortconv_is_block_local(self):
        model = DConvDraftModel(_tiny_config()).eval()
        base_config = _tiny_config(projector_type=None)
        base_config.dflash_config.pop("projector_type")
        base = DFlashDraftModel(base_config).eval()
        base.load_state_dict(model.state_dict(), strict=False)

        batch_size, context_len = 2, 10
        stream_len = model.stream_len
        anchors = torch.tensor([[6], [7]])
        keep = torch.ones_like(anchors, dtype=torch.bool)
        causal = torch.zeros_like(anchors, dtype=torch.bool)
        position_ids = torch.cat(
            [
                torch.arange(context_len)[None].expand(batch_size, -1),
                (
                    anchors.unsqueeze(-1)
                    + torch.arange(-(model.window_len - 1), model.block_size)
                ).view(batch_size, -1),
            ],
            dim=1,
        )
        attention_mask = create_dconv_sdpa_mask(
            anchors,
            keep,
            causal,
            context_len,
            stream_len,
            torch.device("cpu"),
        )
        noise = torch.randn(batch_size, stream_len, 64)
        target = torch.randn(batch_size, context_len, len(model.target_layer_ids) * 64)
        kwargs = dict(
            position_ids=position_ids,
            attention_mask=attention_mask,
            noise_embedding=noise,
            target_hidden=target,
        )
        self.assertTrue(torch.equal(model(**kwargs), base(**kwargs)))

        conv = ShortConv(8, 4)
        conv.kernel.data.normal_()
        inputs = torch.randn(1, 2 * stream_len, 8)
        outputs = conv(inputs, num_blocks=2)
        perturbed = inputs.clone()
        perturbed[:, 2] += 1
        perturbed_outputs = conv(perturbed, num_blocks=2)
        self.assertTrue(torch.equal(outputs[:, :2], perturbed_outputs[:, :2]))
        self.assertTrue(
            torch.equal(outputs[:, stream_len:], perturbed_outputs[:, stream_len:])
        )

    def test_registry_auto_loader_builds_dconv(self):
        config = _tiny_config()
        config.architectures = ["DConvDraftModel"]
        self.assertIn("DConvDraftModel", available_drafts())
        self.assertIsInstance(AutoEagle3DraftModel.from_config(config), DConvDraftModel)

    def test_online_training_has_finite_loss_and_conv_gradients(self):
        model = DConvDraftModel(_tiny_config())
        embedding = nn.Embedding(257, 64)
        lm_head = nn.Linear(64, 257, bias=False)
        online = OnlineDConvModel(
            model,
            lm_head,
            embedding,
            mask_token_id=256,
            attention_backend="sdpa",
            num_anchors=4,
            rho_reread=0.5,
        )
        input_ids = torch.randint(0, 256, (2, 36))
        hidden_states = torch.randn(
            2, 36, len(model.target_layer_ids) * model.config.hidden_size
        )
        loss_mask = torch.ones(2, 36)
        loss_mask[:, :6] = 0

        loss, accuracy, metrics = online(input_ids, hidden_states, loss_mask)
        loss.backward()
        conv_grad = sum(
            module.kernel.grad.abs().sum()
            for module in model.modules()
            if isinstance(module, ShortConv)
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(accuracy))
        self.assertGreater(conv_grad.item(), 0.0)
        self.assertIn("acc_pass1", metrics)
        self.assertIn("acc_pass2", metrics)
        self.assertIn("reread_frac", metrics)

    def test_window_must_cover_every_convolution_tap(self):
        config = _tiny_config()
        config.dflash_config["window_len"] = 3
        with self.assertRaisesRegex(ValueError, "window_len"):
            DConvDraftModel(config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
