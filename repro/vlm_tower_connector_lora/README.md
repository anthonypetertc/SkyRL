# Investigation: full-VLM LoRA and vLLM tower loading

Investigation against SkyRL commit `0cd3ad969e4be707e5cc96f5360b3458e32d7947`
(the fork's `repro/vlm-tower-connector-lora` branch), using vLLM 0.33.0.
The investigation scripts and results live on `investigate/vlm-tower-lora-sync`;
the revision above identifies the code being investigated.
No production fix is included. These scripts record CPU configuration evidence,
pinned Megatron source checks, and actual GPU policy/export/vLLM comparisons
for both FSDP and Megatron.

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

**The visual-adapter omission is reproduced at runtime for both FSDP and
Megatron adapter-only sync on dense Qwen3-VL with standard LoRA.** This does not
mean every Megatron VLM adapts its visual tower by default. The Megatron runtime
results below also show that enabling tower LoRA does not establish policy parity.

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
arguments, this identifies the same missing-flag mechanism for Qwen3-VL
Megatron with `merge_lora=false`. The native runtime experiment below confirms
trainable visual adapters, actual export, and omission by flag-off vLLM.

By contrast, the pinned Qwen2.5-VL bridge uses an HF vision tower. Its typical
visual leaf names (`qkv`, `proj`, `gate_proj`, `up_proj`, `down_proj`, numeric
merger layers) do not match the default Megatron targets (`linear_qkv`,
`linear_proj`, `linear_fc1`, `linear_fc2`, `in_proj`, `out_proj`). The check
covers representative names, not an exhaustive instantiated Qwen2.5-VL model.
Custom targets can change the scope.

`merge_lora=true` uses merged/base-weight sync and does not depend on vLLM's
adapter tower loader. This investigation makes no claim about every aspect
of that separate sync path.

**Source-check boundary:** this script executes no Megatron model. The separate
GPU experiment below supplies native forward/export evidence. The source JSON
records exact source URLs, content hashes, matched names, and resolved export names:
[backend_scope.json](results/backend_scope.json).

## HF/FSDP GPU comparison

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
and no Megatron execution in this script. The measurement uses a genuine pretrained VLM
and real vLLM forward passes; it measures loader correctness rather than an
end-to-end RL training outcome. No inference about training collapse follows.

## Megatron runtime experiment

The companion [measure_megatron_logprobs.py](measure_megatron_logprobs.py) uses
an actual `MegatronPolicyWorkerBase` for checkpoint loading, native policy
forward passes, and `_save_lora_adapters_and_sync`. HF supplies the pinned
checkpoint and input processor; primary policy scores come from Megatron.
Run from the repository root:

```bash
RAY_ADDRESS=local VLLM_WORKER_MULTIPROC_METHOD=spawn \
NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1 NVTE_UNFUSED_ATTN=0 \
uv run --isolated --extra megatron python \
  repro/vlm_tower_connector_lora/measure_megatron_logprobs.py \
  --output-dir repro/vlm_tower_connector_lora/results/megatron_measurement \
  > /tmp/skyrl_megatron_vlm_measurement.log 2>&1
```

This uses TP/PP/CP/EP=1, full-VLM standard LoRA, `merge_lora=false`, BF16, and
inference-only initialization (no DDP buffers or optimizer). It scores the same
image and 12 continuation tokens as the HF/FSDP experiment. Initially zero B
factors provide a baseline. Name-seeded visual factors are then set nonzero,
followed by language factors, with each adapter exported through the actual
Megatron worker and loaded directly into vLLM. Publication uses a recording
sink; it does not exercise HTTP/Ray transport or distributed export.

The script records actual trainable adapter names, shapes, classes and dtypes,
native policy scores, exported tensor nonzero counts, artifact hashes, and
package revisions. It rejects a vision-only export with nonzero language B
factors and rejects nonfinite scores or input alignment failures.

The Platform base image also supplies CUDA 12.6, which Transformer Engine can
find ahead of its installed CUDA-13 runtime compiler. The launcher explicitly
selects the installed CUDA-13 NVRTC, matching builtins and headers before
loading torch/TE, and records the selection and loaded NVRTC library paths.
This repairs the runtime environment without modifying the model or weights.

### Observed Megatron logprob divergence

The native run completed on 2026-11-16 on one H100 80GB, using torch
2.14.0+cu130, transformers 5.16.1, PEFT 0.21.0, vLLM 0.33.0, Transformer
Engine 2.19.0, Megatron-Core `ed9c2c8c1cf4a5438922ae8d215181e7c8b9e0fe`,
and Megatron-Bridge `7171584bd7452caf2adfa6afd12fb89cc37fbc64`.
The checkpoint is explicitly pinned to
`89644892e4d85e24eaac8bacfd4f463576704203` for every stage.

Native initialization produced **432 trainable adapter parameter tensors,
including 208 visual tensors**. Each real Bridge export contains 600 tensors,
including **208 nonzero visual tensors**, all BF16. Splitting native fused
projections accounts for the larger exported tensor count. The vision-only
artifact has **196 language B tensors, all zero**. All six vLLM requests were
accepted, aligned, finite, and scored the same 12 continuation tokens.

| Adapter | Flag off: max absolute Megatron−vLLM | Flag on: max absolute Megatron−vLLM | Off mean | On mean |
|---|---:|---:|---:|---:|
| Base, zero/no adapter | 0.124999 | 0.124999 | 0.033556 | 0.033556 |
| Vision-only native export | 0.377460 | 0.374975 | 0.074628 | 0.075888 |
| Full native export | 0.252431 | 0.623064 | 0.038353 | 0.098140 |

All differences are in nats. **Flag-on worsened the full-adapter maximum and
mean discrepancy in this run; it is not a demonstrated parity fix.** The base
row also has a nonzero discrepancy. Different attention/linear kernels, native
model implementation, internal position construction and tensor casting remain
possible contributors; the table alone does not assign their causes. Processor
tokens, pixels and grids are checked, but that does not establish identical
internal computations in Megatron and vLLM.

The isolated visual control establishes the omission independently of the
cross-engine baseline:

- Native Megatron's vision-only update changes logprobs by up to **0.252461**.
- With the flag **off**, vLLM's vision-only scores equal base **exactly on all
  12 tokens**: maximum visual effect **0.000000**.
- With the flag **on**, the same native-exported artifact changes vLLM scores
  by up to **0.489734**.
- Base vLLM scores are identical across flag settings; the full-adapter scores
  change by up to **0.370633**.

Thus the missing-visual-update mechanism is confirmed for this real Megatron
configuration. The flag activates a visual effect, while agreement with the
native policy remains unresolved. This supports a fail/warn guardrail and
does not support unconditionally auto-enabling the flag as a complete fix.
vLLM emits ignored-visual-module warnings with the flag off.

Evidence: [native policy scores](results/megatron_measurement/policy.json),
[flag off](results/megatron_measurement/off.json),
[flag on](results/megatron_measurement/on.json),
[summary](results/megatron_measurement/summary.json),
[parameter inventory](results/megatron_measurement/megatron_parameters.json),
[export inspection](results/megatron_measurement/megatron_export.json),
[verification](results/megatron_measurement/verification.json), and corresponding
logs. Adapter binaries are excluded from Git and are recreated by the script.

### Separate HF diagnostic of the Megatron artifacts

The optional `--stage hf-export` loads the **unchanged exported tensors** into
HF/PEFT with BF16 adapters (`autocast_adapter_dtype=false`) and eager attention.
It selects exact module paths from the artifact keys **in memory**. Bridge's
exported suffix targets include `proj`, which also matches HF's
`visual.patch_embed.proj`; no corresponding adapter exists in the export.
Loading the unmodified suffix config therefore produced missing adapter keys
in the first diagnostic attempt. Exact-path selection avoids adding that
unexported adapter; it does not validate unmodified-config HF loading.
Neither the files nor the primary experiment are modified.

Both artifacts passed bitwise tensor/dtype round-trip checks with no missing
adapter or unexpected keys. HF used the same pinned checkpoint, tokens, image
grid and normalized pixel hash. Maximum absolute logprob errors, in nats:

| Adapter | Native Megatron−HF artifact | HF artifact−vLLM off | HF artifact−vLLM on |
|---|---:|---:|---:|
| Base | 0.244904 | 0.366848 | 0.366848 |
| Vision-only | 0.336948 | 0.371655 | 0.124998 |
| Full | 0.498013 | 0.390380 | 0.245493 |

This diagnostic also lacks native-policy parity. Flag-on is closer to HF for
these adapters, while the primary native Megatron comparison becomes worse
for the full adapter. These observations cannot distinguish a conversion
problem from execution or numerical differences. No specific cause for the
remaining mismatch is established here.

Evidence: [HF diagnostic](results/megatron_measurement/hf_export_diagnostic.json),
[diagnostic log](results/megatron_measurement/hf_export_diagnostic.log), and
[execution provenance](results/megatron_measurement/execution.json).

**Scope:** a real single-rank Megatron forward/export/vLLM loader experiment
for dense Qwen3-VL, full-VLM default LoRA, `merge_lora=false`. There is no
optimizer step, DDP wrapping, multi-rank export, or remote publication transport.
One image, one fixed continuation and seeded factors do not characterize the
frequency or magnitude of drift in trained policies or other architectures.

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
