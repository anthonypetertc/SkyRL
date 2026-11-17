#!/usr/bin/env python3
"""Original lightweight FSDP configuration probe for VLM tower/connector LoRA.

This checks SkyRL engine arguments and PEFT keys on a stand-in VLM. It does
not run a vLLM forward and cannot establish whether an adapter is ignored,
rejected, or warned about. See cpu_check.py for the canonical reproduction
using a real tiny HF VLM, README.md for CPU instructions, and measure_logprobs.py
for the FSDP-export policy/rollout comparison on GPU.

Run from the repository root:
    CUDA_VISIBLE_DEVICES='' uv run --isolated --extra fsdp python \
        repro/vlm_tower_connector_lora/repro.py --expect buggy

Here --expect buggy/fixed means only that the tower flag is absent/present.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="A vLLM tower/connector-LoRA-capable VLM checkpoint id (never downloaded; used as a string).",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument(
        "--expect",
        choices=["buggy", "fixed", "none"],
        default="buggy",
        help="Assert the tower flag is missing (buggy), present (fixed), or just report (none).",
    )
    return parser.parse_args()


def build_engine_args(
    model: str,
    lora_rank: int,
    *,
    language_model_only: bool = False,
    tower_lora: Optional[bool] = None,
):
    """Build the vLLM engine args SkyRL would launch the policy with."""
    from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args
    from skyrl.train.config import SkyRLTrainConfig

    cfg = SkyRLTrainConfig.from_cli_overrides(
        [
            "trainer.strategy=fsdp",
            f"trainer.policy.model.path={model}",
            f"trainer.policy.model.lora.rank={lora_rank}",
            f"trainer.policy.language_model_only={str(language_model_only).lower()}",
            f"trainer.ref.language_model_only={str(language_model_only).lower()}",
            f"generator.inference_engine.language_model_only={str(language_model_only).lower()}",
        ]
    )
    if tower_lora is not None:
        cfg.generator.inference_engine.engine_init_kwargs = {
            **cfg.generator.inference_engine.engine_init_kwargs,
            "enable_tower_connector_lora": tower_lora,
        }
    return build_vllm_cli_args(cfg)


def tower_flag(args) -> bool:
    """vLLM's ``LoRAConfig.enable_tower_connector_lora``; SkyRL only sets it via engine_init_kwargs."""
    return bool(getattr(args, "enable_tower_connector_lora", False))


def tower_lora_keys(language_model_only: bool) -> list[str]:
    """Vision-tower/connector LoRA keys a full-VLM PEFT adapter carries.

    A stand-in for a real VLM: PEFT's ``"all-linear"`` targets every ``nn.Linear``,
    exactly as SkyRL's default ``target_modules`` does on a full VLM.
    """
    from types import SimpleNamespace

    import torch.nn as nn
    from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict

    class Block(nn.Module):
        def __init__(self, dim: int = 8) -> None:
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(dim, dim)
            self.mlp = nn.Module()
            self.mlp.gate_proj = nn.Linear(dim, dim)

    class TinyVLM(nn.Module):
        # Enough of a transformers surface for PEFT's causal-LM wrapper.
        base_model_prefix = "model"
        config = SimpleNamespace(
            use_cache=False,
            is_encoder_decoder=False,
            pad_token_id=0,
            model_type="tiny",
        )

        def prepare_inputs_for_generation(self, *args, **kwargs):
            return {}

        def __init__(self, dim: int = 8) -> None:
            super().__init__()
            self.visual = nn.Module()
            self.visual.blocks = nn.ModuleList([Block(dim)])  # vision tower
            self.visual.merger = nn.Module()
            self.visual.merger.mlp = nn.ModuleList([nn.Linear(dim, dim)])  # connector
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Block(dim)])  # language model

    model = TinyVLM()
    if language_model_only:
        # SkyRL loads only the text backbone in this mode, so the tower is never adapted.
        del model.visual
    peft_model = get_peft_model(
        model,
        LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", task_type=TaskType.CAUSAL_LM),
    )
    state = get_peft_model_state_dict(peft_model)
    return sorted(name for name in state if "lora_" in name and ".visual." in name)


def main() -> int:
    args = parse_args()

    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import vllm  # noqa: F401
    except ImportError as error:
        print(f"ERROR: missing dependency: {error}", file=sys.stderr)
        print(
            "Run inside SkyRL's environment, e.g. "
            "`uv run --isolated --extra fsdp python repro/vlm_tower_connector_lora/repro.py`.",
            file=sys.stderr,
        )
        return 2

    from vllm.config import LoRAConfig

    vllm_default = LoRAConfig().enable_tower_connector_lora
    print("vLLM gates tower/connector LoRA behind a config switch:")
    print(f"  vllm LoRAConfig().enable_tower_connector_lora default = {vllm_default}")

    full_args = build_engine_args(args.model, args.lora_rank, language_model_only=False)
    lmo_args = build_engine_args(args.model, args.lora_rank, language_model_only=True)
    fixed_args = build_engine_args(args.model, args.lora_rank, language_model_only=False, tower_lora=True)

    try:
        full_keys = tower_lora_keys(language_model_only=False)
        lmo_keys = tower_lora_keys(language_model_only=True)
        keys_error: Optional[Exception] = None
    except Exception as error:  # pragma: no cover - PEFT should work, but do not hide the engine check
        full_keys, lmo_keys, keys_error = [], [], error

    print()
    print(f"model under test: {args.model}  (lora_rank={args.lora_rank})")
    if keys_error is not None:
        print(f"  [warn] could not build the stand-in PEFT adapter: {keys_error}")
    else:
        print(f"  trainer adapter tower/connector keys (full VLM):            {len(full_keys)}")
        for key in full_keys[:6]:
            print(f"      {key}")
        print(f"  trainer adapter tower/connector keys (language_model_only): {len(lmo_keys)}")

    print()
    print(f"{'case':<56}{'enable_lora':>12}{'tower_lora':>12}")
    print(f"{'full VLM + LoRA (default)':<56}" f"{str(full_args.enable_lora):>12}{str(tower_flag(full_args)):>12}")
    print(
        f"{'language_model_only VLM + LoRA (control)':<56}"
        f"{str(lmo_args.enable_lora):>12}{str(tower_flag(lmo_args)):>12}"
    )
    print(
        f"{'full VLM + LoRA + engine_init_kwargs override':<56}"
        f"{str(fixed_args.enable_lora):>12}{str(tower_flag(fixed_args)):>12}"
    )

    failures: list[str] = []
    if keys_error is None:
        if not full_keys:
            failures.append("the full-VLM adapter has no visual.* keys; the stand-in is not exercising the bug")
        if lmo_keys:
            failures.append("the language_model_only adapter unexpectedly has visual.* keys")

    if args.expect == "buggy":
        if not full_args.enable_lora:
            failures.append("SkyRL did not enable LoRA on the engine, so the repro is not in the LoRA path")
        if tower_flag(full_args):
            failures.append("SkyRL enables tower/connector LoRA; expected the flag to be absent")
        if vllm_default:
            failures.append("vLLM now defaults enable_tower_connector_lora to True; expected False")
    elif args.expect == "fixed":
        if not tower_flag(full_args):
            failures.append("SkyRL still does not enable tower/connector LoRA for a full VLM + LoRA")
    if not tower_flag(fixed_args):
        failures.append("engine_init_kwargs.enable_tower_connector_lora did not reach the engine args")

    print()
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    if args.expect == "buggy":
        print("OK: the full-VLM stand-in has visual adapter keys, but the engine tower flag is False.")
        print("    This configuration probe does not execute vLLM or verify loader behavior.")
    elif args.expect == "fixed":
        print("OK: full VLM + LoRA enables tower/connector LoRA on the engine.")
    else:
        print("OK: reported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
