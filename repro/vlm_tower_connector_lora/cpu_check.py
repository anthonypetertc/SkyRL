"""CPU-only check of SkyRL's FSDP VLM LoRA targets and vLLM arguments.

Run from the repository root with CUDA_VISIBLE_DEVICES='' and the fsdp extra.
No distributed process group, Ray actors, model download, or CUDA tensor is used.
This checks configuration and adapter contents; it does not measure logprobs.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args
from skyrl.backends.skyrl_train.utils.lora_targets import resolve_lora_target_modules
from skyrl.train.config import SkyRLTrainConfig


def tiny_model():
    """A real HF Qwen3-VL architecture with deliberately small random weights."""
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 256,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "out_hidden_size": 64,
            "deepstack_visual_indexes": [0, 1],
        },
        image_token_id=250,
        video_token_id=251,
        vision_start_token_id=252,
        vision_end_token_id=253,
    )
    return Qwen3VLForConditionalGeneration(config).cpu()


def adapter_keys(language_only):
    model = tiny_model()
    if language_only:
        # The actual text submodel, with no visual tower in its module tree.
        model = model.model.language_model
    targets = resolve_lora_target_modules(model, "all-linear", train_attn=True, train_mlp=True)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=8, target_modules=targets))
    state = get_peft_model_state_dict(model)
    assert all(tensor.device.type == "cpu" for tensor in state.values())
    return sorted(state)


def engine_case(language_only, override=None):
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "fsdp"
    cfg.trainer.policy.model.path = "Qwen/Qwen3-VL-2B-Instruct"
    cfg.trainer.policy.model.lora.rank = 8
    cfg.trainer.policy.language_model_only = language_only
    cfg.generator.inference_engine.language_model_only = language_only
    if override is not None:
        cfg.generator.inference_engine.engine_init_kwargs = {"enable_tower_connector_lora": override}
    args = build_vllm_cli_args(cfg)
    return {
        "strategy": cfg.trainer.strategy,
        "language_model_only": language_only,
        "override": override,
        "enable_lora": args.enable_lora,
        "enable_tower_connector_lora": args.enable_tower_connector_lora,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("cpu_check.json"))
    args = parser.parse_args()
    assert not torch.cuda.is_available(), "Run with CUDA_VISIBLE_DEVICES='' to enforce the CPU-only check"
    from vllm.config import LoRAConfig

    full_keys, text_keys = adapter_keys(False), adapter_keys(True)
    visual_keys = [key for key in full_keys if ".visual." in key]
    text_visual_keys = [key for key in text_keys if ".visual." in key]
    assert visual_keys and not text_visual_keys
    cases = [engine_case(language_only, override) for language_only in [False, True] for override in [None, True]]
    assert LoRAConfig().enable_tower_connector_lora is False
    for case in cases:
        assert case["enable_lora"] is True
        assert case["enable_tower_connector_lora"] == (case["override"] is True)
    result = {
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "peft", "transformers", "vllm")},
        "cuda_available": torch.cuda.is_available(),
        "visual_adapter_tensor_count": len(visual_keys),
        "full_adapter_tensor_count": len(full_keys),
        "text_only_visual_tensor_count": len(text_visual_keys),
        "visual_adapter_keys": visual_keys,
        "cases": cases,
        "scope": "FSDP configuration and actual HF/PEFT target selection; no distributed worker or vLLM forward.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
