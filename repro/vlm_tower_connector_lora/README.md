# Full-VLM LoRA reproduction

## CPU reproduction: observed

[cpu_check.py](cpu_check.py) checks SkyRL's LoRA targets and vLLM arguments
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
- FSDP and Megatron adapter-only configurations enable LoRA but leave
  `enable_tower_connector_lora=false`.

This is a configuration check; it does not run vLLM inference or a Megatron model.

[Full investigation and GPU results](INVESTIGATION.md).
