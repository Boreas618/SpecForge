import gc
import glob
import json
import os
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig


class _RawConfigShim:
    """Attribute view over a raw config.json, for checkpoints whose model_type
    transformers does not recognize and that ship no remote config code (e.g.
    NDA/Inkling's ``inkling_mm_model``). Nested dicts become shims recursively;
    missing keys raise AttributeError so ``hasattr``/``getattr`` defaults work."""

    def __init__(self, data: dict):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name):
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(name) from None
        return _RawConfigShim(value) if isinstance(value, dict) else value


def load_target_config(
    model_path: str,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
):
    """AutoConfig with a raw-config.json fallback for unknown model types."""
    try:
        return AutoConfig.from_pretrained(
            model_path, cache_dir=cache_dir, trust_remote_code=trust_remote_code
        )
    except (ValueError, KeyError, OSError) as exc:
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise
        with open(config_path, "r") as f:
            raw = json.load(f)
        print(
            f"[TargetEmbeddingsAndHead] AutoConfig failed "
            f"({type(exc).__name__}); using raw config.json shim for "
            f"model_type={raw.get('model_type')!r}"
        )
        return _RawConfigShim(raw)


class TargetEmbeddingsAndHead(nn.Module):
    """
    Efficiently loads only the embedding layer and lm_head from a pretrained model.
    Handles safetensors slicing and Weight Tying correctly.
    """

    @staticmethod
    def _checkpoint_vocab_size(cfg):
        # The embed/unembed tensors in the checkpoint span the PADDED vocab.
        # Registered config classes may rewrite vocab_size to the unpadded
        # size for sampling (the nda_sgl fork's InklingConfig does: 200058
        # vs padded 201024), so prefer padded_vocab_size when present. The
        # raw-json shim path has no padded_vocab_size and vocab_size is
        # already the padded value.
        return getattr(cfg, "padded_vocab_size", None) or cfg.vocab_size

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Support for MLLMs with separate text_config
        if hasattr(config, "text_config"):
            vocab = self._checkpoint_vocab_size(config.text_config)
            self.embed_tokens = nn.Embedding(
                vocab,
                config.text_config.hidden_size,
                padding_idx=getattr(config.text_config, "pad_token_id", None),
            )
            self.lm_head = nn.Linear(
                config.text_config.hidden_size,
                vocab,
                bias=False,
            )
        else:
            vocab = self._checkpoint_vocab_size(config)
            self.embed_tokens = nn.Embedding(
                vocab,
                config.hidden_size,
                padding_idx=getattr(config, "pad_token_id", None),
            )
            self.lm_head = nn.Linear(config.hidden_size, vocab, bias=False)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        embed_key: Optional[str] = None,
        lm_head_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
    ) -> "TargetEmbeddingsAndHead":

        # 1. Load Config (AutoConfig, with a raw-json fallback for model types
        # transformers does not know — e.g. NDA/Inkling `inkling_mm_model`).
        config = load_target_config(
            model_path, cache_dir=cache_dir, trust_remote_code=trust_remote_code
        )
        instance = cls(config)

        if embed_key is None:
            embed_key = "model.embed_tokens.weight"
        if lm_head_key is None:
            lm_head_key = "lm_head.weight"

        # 2. Resolve Model Path
        local_model_path = model_path
        if not os.path.exists(local_model_path):
            try:
                local_model_path = snapshot_download(
                    repo_id=model_path,
                    cache_dir=cache_dir,
                    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model"],
                )
            except Exception as e:
                print(f"Warning: Snapshot download failed or path check failed: {e}")

        # 3. Handle Weight Tying
        tie_weights = getattr(config, "tie_word_embeddings", False)

        # 4. Load Weights
        instance._load_weights(local_model_path, embed_key, lm_head_key, tie_weights)

        # 4b. muP logit-scale fold. Some targets (NDA/Inkling) compute
        # logits = W_vocab · (H / mup) rather than W_vocab · H; every consumer
        # of this frozen head (DSpark CE/L1/confidence objectives, the DeepSpec
        # accept-length verifier at any temperature) needs the TRUE target
        # distribution, so fold 1/mup into the weight copy exactly once here.
        # Greedy argmax is unaffected; softmax temperature is corrected.
        mup = getattr(config, "logits_mup_width_multiplier", None)
        if mup is None and hasattr(config, "text_config"):
            mup = getattr(config.text_config, "logits_mup_width_multiplier", None)
        if mup:
            if tie_weights:
                raise RuntimeError(
                    "logits_mup_width_multiplier with tied embeddings would "
                    "corrupt the embedding table when folding the head scale; "
                    "refusing. Untie or handle the scale explicitly."
                )
            instance.lm_head.weight.data.div_(float(mup))
            instance.lm_head_mup_folded = float(mup)
            print(
                f"[TargetEmbeddingsAndHead] folded 1/{mup} muP logit scale into "
                f"the frozen lm_head copy (logits now match the target's true "
                f"distribution at any temperature)"
            )

        # 5. Move to Device & Freeze
        instance.to(device=device, dtype=dtype)
        instance.eval()
        instance.requires_grad_(False)

        return instance

    def _load_weights(
        self, model_path: str, embed_key: str, lm_head_key: str, tie_weights: bool
    ):
        index_files = glob.glob(os.path.join(model_path, "*.index.json"))
        weight_map = {}
        files_to_load = {}

        if index_files:
            with open(index_files[0], "r") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})

            if embed_key in weight_map:
                files_to_load[embed_key] = weight_map[embed_key]
            else:
                raise ValueError(
                    f"Embedding key '{embed_key}' not found in weight map."
                )

            if not tie_weights:
                if lm_head_key in weight_map:
                    files_to_load[lm_head_key] = weight_map[lm_head_key]
                else:
                    print(
                        f"Warning: {lm_head_key} not found. Ensure model doesn't use tied weights manually."
                    )
        else:
            safetensors = glob.glob(os.path.join(model_path, "*.safetensors"))
            bins = glob.glob(os.path.join(model_path, "*.bin"))
            target_file = safetensors[0] if safetensors else (bins[0] if bins else None)

            if not target_file:
                raise FileNotFoundError("No checkpoint found.")

            files_to_load[embed_key] = os.path.basename(target_file)
            if not tie_weights:
                files_to_load[lm_head_key] = os.path.basename(target_file)

        loaded_keys = set()

        file_to_keys_map = {}
        for key, filename in files_to_load.items():
            full_path = os.path.join(model_path, filename)
            if full_path not in file_to_keys_map:
                file_to_keys_map[full_path] = []
            file_to_keys_map[full_path].append(key)

        for file_path, keys in file_to_keys_map.items():
            self._load_file_content(file_path, keys, embed_key, lm_head_key)
            loaded_keys.update(keys)

        if tie_weights:
            print(
                "Weight tying detected: Sharing weights between Embeddings and LM Head."
            )
            self.lm_head.weight = self.embed_tokens.weight

        if embed_key not in loaded_keys:
            raise RuntimeError("Failed to load embeddings.")
        if not tie_weights and lm_head_key not in loaded_keys:
            print(
                "Warning: LM Head weights were not found (and tie_weights is False). Head is random."
            )

    def _load_file_content(
        self,
        file_path: str,
        keys_to_extract: list,
        target_embed_key: str,
        target_head_key: str,
    ):
        """Helper to load specific keys from a file"""
        print(f"Loading {keys_to_extract} from {os.path.basename(file_path)}...")

        state_dict_part = {}

        if file_path.endswith(".safetensors"):
            with safe_open(file_path, framework="pt") as f:
                for k in keys_to_extract:
                    if k in f.keys():
                        state_dict_part[k] = f.get_tensor(k)
        else:
            print(
                f"Warning: Loading .bin file {os.path.basename(file_path)} into RAM. Convert to safetensors for efficiency."
            )
            full_state = torch.load(file_path, map_location="cpu")
            for k in keys_to_extract:
                if k in full_state:
                    state_dict_part[k] = full_state[k]
            del full_state
            gc.collect()

        for k, tensor in state_dict_part.items():
            if k == target_embed_key:
                self.embed_tokens.weight.data.copy_(tensor)
                print(" -> Loaded Embeddings")
            elif k == target_head_key:
                if tensor.shape == self.lm_head.weight.data.shape:
                    self.lm_head.weight.data.copy_(tensor)
                    print(" -> Loaded LM Head")
                else:
                    raise RuntimeError(
                        f"Shape mismatch for {k}. Expected {self.lm_head.weight.shape}, got {tensor.shape}"
                    )
