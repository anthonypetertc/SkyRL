# FSDP VLM LoRA reproduction

[cpu_check.py](cpu_check.py) checks SkyRL's FSDP LoRA targets and vLLM arguments
using a tiny, randomly initialized Qwen3-VL. No GPU or model download is needed.

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' uv run --isolated --extra fsdp python \
  repro/vlm_tower_connector_lora/cpu_check.py \
  --output /tmp/skyrl_vlm_cpu.json \
  > /tmp/skyrl_vlm_cpu.log 2>&1
```

Observed at the affected revision with vLLM 0.33.0:

- Full VLM: **28 visual adapter tensors**; language-only control: **zero**.
- FSDP enables rollout LoRA but leaves `enable_tower_connector_lora=false`.
- An explicit engine override sets the tower flag to `true`.

This checks configuration and adapter contents; it does not run distributed FSDP
or vLLM inference.
