# Minimal repro: VLM + LoRA drops the vision-tower adapter in vLLM

A self-contained, GPU-less repro for the bug where a full VLM policy trained with
FSDP + LoRA has its vision-tower/connector adapter weights silently ignored by the
vLLM rollout engine.

## The bug in one paragraph

SkyRL's FSDP + LoRA path applies LoRA to every linear layer, including the vision
tower and connector (`target_modules="all-linear"`), and exports those `visual.*`
weights in the PEFT adapter it hot-loads into vLLM. But `build_vllm_cli_args`
(`skyrl/backends/skyrl_train/inference_servers/utils.py`) never sets vLLM's
`enable_tower_connector_lora`, which defaults to `False`. vLLM's LoRA manager
(`vllm/lora/model_manager.py::_maybe_init_mm`) then sets
`supports_tower_connector_lora = False` and returns before building the tower punica
wrapper, so `load_lora_adapter` drops the `visual.*` weights. The policy optimizes a
trained vision tower while the rollout engine runs the frozen base tower, so rollout
logprobs / KL / rewards are computed against a different model.

## What the script checks

- **Trainer export**: a LoRA on a stand-in full VLM produces `visual.*` tower/connector
  keys; a `language_model_only` VLM produces none.
- **Engine args**: `build_vllm_cli_args` leaves `enable_tower_connector_lora` unset
  (vLLM default `False`) for a full VLM + LoRA config, even with `enable_lora=True`.
- **Workaround**: setting `generator.inference_engine.engine_init_kwargs.enable_tower_connector_lora=true`
  flips it on.

The trainer export side is already covered upstream by
`tests/backends/skyrl_train/workers/test_lora_export_names.py`; this script ties it to
the engine args that drop the weights.

## Run

Needs SkyRL's deps (torch, peft, vllm) but **no GPU** and no model download:

```bash
uv run --isolated --extra fsdp python repro/vlm_tower_connector_lora/repro.py --expect buggy
```

Or, if SkyRL + vLLM are already importable:

```bash
python repro/vlm_tower_connector_lora/repro.py --expect buggy
```

`--expect buggy` asserts the flag is missing (reproduces the bug);
`--expect fixed` asserts it is present (regression check once patched);
`--expect none` just reports. Exit code is `0` on match, `1` otherwise.

## Suggested upstream fix

In `build_vllm_cli_args`, when `_uses_lora_weight_sync(cfg)` is true and the served
model is a multimodal VLM that is not `language_model_only`, set
`args.enable_tower_connector_lora = True` (vLLM only supports this for some MM
architectures, so a capability check or a loud warning is appropriate). The detection
helper already exists: `skyrl.utils.tok.check_is_vlm`.
