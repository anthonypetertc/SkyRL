# Investigation: full-VLM LoRA and vLLM tower loading

Investigation against SkyRL commit `0cd3ad969e4be707e5cc96f5360b3458e32d7947`
(the fork's `repro/vlm-tower-connector-lora` branch), using vLLM 0.33.0.
No production fix is included. These scripts separate configuration evidence,
Megatron source evidence, and an actual GPU loader/forward comparison.

## CPU reproduction: observed

Run from the repository root, with SkyRL's FSDP dependencies installed:

```bash
CUDA_VISIBLE_DEVICES='' uv run --isolated --extra fsdp python \
  repro/vlm_tower_connector_lora/cpu_check.py \
  --output repro/vlm_tower_connector_lora/results/cpu.json \
  > /tmp/skyrl_vlm_cpu.log 2>&1
```

This uses a real, randomly initialized tiny `Qwen3VLForConditionalGeneration`
on CPU, PEFT, SkyRL's target resolver, and `build_vllm_cli_args`. No checkpoint
is downloaded, no CUDA tensors are created, and no distributed actors start.
The recorded run had `torch.cuda.is_available() == False`.

- Full VLM: 56 adapter tensors, including **28 visual tensors** in the tower,
  main merger, and deepstack mergers.
- Language-only control: **zero visual adapter tensors**.
- SkyRL's engine-argument construction produces the following defaults:

| Trainer strategy | Megatron `merge_lora` | `enable_lora` | `enable_tower_connector_lora` |
|---|---:|---:|---:|
| FSDP | n/a | true | false |
| Megatron | true | false | false |
| Megatron | false | true | false |

The argument check covers both values of `language_model_only` and the explicit
`engine_init_kwargs.enable_tower_connector_lora=true` override: 12 cases total.
The override reaches vLLM unchanged. The Megatron rows test argument construction;
they do not instantiate a Megatron policy.

Evidence: [CPU JSON](results/cpu.json), [CPU log](results/cpu.log).
The original stand-in-model script, [repro.py](repro.py), remains as a smaller
configuration probe; its `--expect buggy` assertion means the flag is absent,
not that a forward pass or silent failure has been demonstrated.

## Backend scope: architecture matters

**FSDP is exposed. Megatron adapter-only sync is also exposed at the source level
for dense Qwen3-VL with standard LoRA; calling this universally FSDP-only is too narrow.** This does not
mean every Megatron VLM adapts its visual tower by default.

The source-only check needs just Python's standard library and public network
access. It fetches Megatron-Bridge at SkyRL's exact pinned revision
`7171584bd7452caf2adfa6afd12fb89cc37fbc64`:

```bash
uv run --isolated --no-project --no-config python \
  repro/vlm_tower_connector_lora/backend_scope.py \
  --output repro/vlm_tower_connector_lora/results/backend_scope.json \
  > /tmp/skyrl_vlm_scope.log 2>&1
```

It executes the pinned target matcher and HF adapter-name resolver verbatim,
using the real Qwen3-VL bridge mappings. Eight visual module patterns match
SkyRL's default Megatron LoRA targets: attention QKV/output, MLP fc1/fc2,
deepstack merger fc1/fc2, and main merger fc1/fc2. Their adapter names resolve
to `model.visual.*.lora_A/B.weight`.

The dense Qwen3-VL provider defaults both vision-freezing flags to false.
SkyRL constructs `SkyRLLoRA(LoRA)`, which adds no vision-freezing override;
it does not construct Bridge's separate `VLMLoRA` class. Its adapter-only
sync path invokes `bridge.export_adapter_weights` and publishes the resulting
adapter to vLLM without a visual-key filter. Combined with the observed engine
arguments, this establishes the same missing-flag mechanism for Qwen3-VL
Megatron with `merge_lora=false`, subject to successful runtime export.

By contrast, the pinned Qwen2.5-VL bridge uses an HF vision tower. Its typical
visual leaf names (`qkv`, `proj`, `gate_proj`, `up_proj`, `down_proj`, numeric
merger layers) do not match the default Megatron targets (`linear_qkv`,
`linear_proj`, `linear_fc1`, `linear_fc2`, `in_proj`, `out_proj`). The check
covers representative names, not an exhaustive instantiated Qwen2.5-VL model.
Custom targets can change the scope.

`merge_lora=true` uses merged/base-weight sync and does not depend on vLLM's
adapter tower loader. This investigation makes no claim about every aspect
of that separate sync path.

**Evidence boundary:** no Megatron policy, optimizer, distributed collective,
or runtime Megatron export was executed. The JSON records exact source URLs,
content hashes, matched names, and resolved export names:
[backend_scope.json](results/backend_scope.json).

## GPU comparison

[measure_logprobs.py](measure_logprobs.py) runs a pretrained
`Qwen/Qwen3-VL-2B-Instruct` policy and vLLM in separate processes on one GPU.
It scores the same 12 fixed continuation tokens on a generated 128-by-128 image.
The model revision is pinned to `89644892e4d85e24eaac8bacfd4f463576704203`.
The original policy run resolved `main`; the public model's last modification
was 2025-10-23, and later rollout runs explicitly pin that revision.
Before reporting results, it requires exact equality of expanded token IDs,
BF16 pixel hashes, and image grids between HF and vLLM. The harness sets
`mm_device_do_normalize=false` so both processors expose normalized CPU pixels
for comparison. vLLM 0.33 normally retains uint8 CPU pixels and normalizes on
the GPU; directly comparing that intermediate representation with HF's
normalized tensor is invalid. The harness observes processor outputs without
substituting pixels or embeddings.

The adapter is deliberately initialized with seeded nonzero factors rather
than trained: rank 8, alpha 8, dropout 0, seed 7819, visual factor standard
deviation 0.02 and language factor standard deviation 0.01. This is a controlled
adapter-loading experiment, not a claim about typical trained-adapter severity.
It includes a vision-only case whose language LoRA B factors remain zero,
plus a full adapter. The full adapter is exported by the actual
`FSDPPolicyWorkerBase._save_lora_adapters_and_sync` method, with rank/barrier and
the remote-client transport mocked for the single-device experiment.

```bash
RAY_ADDRESS=local VLLM_WORKER_MULTIPROC_METHOD=spawn \
uv run --isolated --extra fsdp python \
  repro/vlm_tower_connector_lora/measure_logprobs.py \
  --output-dir repro/vlm_tower_connector_lora/results/measurement \
  > /tmp/skyrl_vlm_measurement.log 2>&1
```

Stages `policy`, `off`, `on`, and `summarize` can be invoked separately with
`--stage`; policy artifacts must exist before off/on. Only `summarize` is
stdlib-only. Engine initialization or alignment failures make the summary
incomplete and the controller fail, rather than producing a divergence number.

### Observed logprob divergence

The normalized-input run completed on 2026-11-14 on an H100 MIG 2g.20gb slice.
Versions: torch 2.14.0+cu130, transformers 5.16.1, PEFT 0.21.0, vLLM 0.33.0.
All six requests were accepted and passed the token/grid/pixel alignment checks.
The exported full adapter contains 600 tensors, including **208 visual tensors**.

Numbers are natural-log probability differences (nats), over the same 12
continuation tokens. HF uses eager attention; vLLM uses its selected
FlashAttention 3 kernels. The base-model row supplies the measured numerical
reference; the flag-on error should not be described as zero.

| Adapter | Flag off: max absolute HF−vLLM | Flag on: max absolute HF−vLLM | Off mean | On mean |
|---|---:|---:|---:|---:|
| Base, no adapter | 0.252429 | 0.252429 | 0.045385 | 0.045385 |
| Vision-only nonzero adapter | 0.369358 | 0.112952 | 0.060012 | 0.011301 |
| Full adapter, SkyRL FSDP export | 0.737801 | 0.249907 | 0.089325 | 0.037862 |

The full-adapter maximum mismatch falls from **0.737801 to 0.249907 nats**,
near the **0.252429** base-model discrepancy. This demonstrates an improvement,
not bitwise HF/vLLM parity. PEFT's exported adapter tensors are FP32; vLLM's
default LoRA dtype follows the BF16 model, which is another possible source of
residual differences alongside different attention and linear kernels.

The stronger causal control compares models within vLLM:

- With the flag **off**, the vision-only adapter's 12 logprobs equal vLLM's base
  logprobs **exactly**: max absolute adapter effect is **0.000000**.
- With the flag **on**, that adapter changes vLLM logprobs by up to **0.380656**.
- HF's own vision-only adapter effect is **0.614262**. This establishes that the
  visual adapter is nonzero and behaviorally relevant.
- Toggling the flag changes the full-adapter vLLM logprobs by up to **0.614263**;
  toggling it changes the no-adapter base case by **0.000000**.

Encoder-cache reuse is not the explanation: vLLM 0.33's
`InputProcessor._get_mm_identifier` includes the LoRA name when tower LoRA is
enabled, and the cases use distinct names. Prefix and processor caches are also
disabled.

Evidence: [summary.json](results/measurement/summary.json),
[policy scores and inputs](results/measurement/policy.json),
[flag off](results/measurement/off.json), [flag on](results/measurement/on.json),
[FSDP export inspection](results/measurement/fsdp_export.json), and corresponding
`.log` files. Adapter binaries are deliberately excluded from Git; the seeded
script recreates them.

**Limitations:** one image and continuation, deliberately seeded nonzero factors,
no optimizer step, no distributed FSDP sharding or real HTTP/Ray sync transport,
and no Megatron runtime export. The measurement uses a genuine pretrained VLM
and real vLLM forward passes; it measures loader correctness rather than an
end-to-end RL training outcome. No inference about training collapse follows.

## Correctness framing and candidate guardrail

The CPU result proves a default/configuration gap. It alone cannot prove
whether vLLM silently drops, rejects, or warns about visual adapter weights.
In the actual vLLM 0.33.0 run, flag-off initialization logs **warnings**, e.g.:

```text
no matching PunicaWrapper is found; visual.merger.linear_fc1 will be ignored.
```

Therefore, “no error and no warning” is inaccurate for this version. A useful
SkyRL guardrail would inspect exported visual adapter keys and reject or clearly
warn when the inference engine has tower LoRA disabled. Startup validation could
catch supported default configurations earlier. Automatically enabling the flag
requires model capability and vLLM correctness considerations; the explicit
engine-kwargs override is a configuration workaround, not evidence that all
supported architectures work correctly with the feature enabled.

Relevant SkyRL paths at the investigated revision:

- `skyrl/backends/skyrl_train/utils/lora_targets.py`: PEFT and Megatron targets.
- `skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py`: full-policy export.
- `skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py`:
  `configure_lora`, `_save_lora_adapters_and_sync`, and adapter-only dispatch.
- `skyrl/backends/skyrl_train/inference_servers/utils.py`:
  `_uses_lora_weight_sync`, `build_vllm_cli_args`, and engine-kwargs pass-through.

No issue or pull request was created.
