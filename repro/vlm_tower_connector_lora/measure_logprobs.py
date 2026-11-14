"""Measure HF policy vs vLLM logprobs on identical image/continuation tokens.

The controller runs each engine in its own process to release GPU memory between
cases. Default: a pretrained Qwen3-VL-2B, one deterministic image and 12 fixed
answer tokens. This is a loader experiment, not an RL training benchmark.
"""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import traceback
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "policy", "off", "on", "summarize"], default="all")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--revision", default="89644892e4d85e24eaac8bacfd4f463576704203")
    parser.add_argument("--output-dir", type=Path, default=Path("vlm_lora_measurement"))
    return parser.parse_args()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare_policy(args):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
    from PIL import Image, ImageDraw
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from skyrl.backends.skyrl_train.utils.lora_targets import (
        resolve_lora_target_modules,
    )

    torch.manual_seed(7819)
    torch.set_grad_enabled(False)
    image = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 16, 60, 112), fill="red")
    draw.ellipse((68, 32, 120, 96), fill="blue")
    image.save(args.output_dir / "prompt.png")
    processor = AutoProcessor.from_pretrained(
        args.model, revision=args.revision, min_pixels=128 * 128, max_pixels=128 * 128
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe the colored shapes in this image briefly."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    continuation = "There is a red rectangle and a blue circle on a white background."
    answer_ids = processor.tokenizer.encode(continuation, add_special_tokens=False)[:12]
    prompt_ids = inputs["input_ids"][0].tolist()
    full_ids = prompt_ids + answer_ids
    inputs["input_ids"] = torch.tensor([full_ids])
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    # Transformers 5's processor supplies per-token modality IDs. The fixed
    # continuation consists of text, whose modality ID is zero.
    for key in ("mm_token_type_ids", "token_type_ids"):
        if key in inputs:
            inputs[key] = torch.cat([inputs[key], torch.zeros((1, len(answer_ids)), dtype=inputs[key].dtype)], dim=1)
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, attn_implementation="eager"
    )
    model = model.to("cuda").eval()

    def score():
        logits = model(**inputs, use_cache=False).logits[0]
        positions = torch.arange(len(prompt_ids) - 1, len(full_ids) - 1, device="cuda")
        answer = torch.tensor(answer_ids, device="cuda")
        return logits[positions].float().log_softmax(-1).gather(1, answer[:, None]).flatten().cpu().tolist()

    base = score()
    targets = resolve_lora_target_modules(model, "all-linear", True, True)
    model = get_peft_model(
        model, LoraConfig(r=8, lora_alpha=8, lora_dropout=0, target_modules=targets, task_type=TaskType.CAUSAL_LM)
    ).eval()
    # Deterministic nonzero visual update. Keep language adapters at zero first,
    # so the off/on comparison isolates the image path.
    for name, parameter in model.named_parameters():
        if ".visual." in name and (".lora_A." in name or ".lora_B." in name):
            parameter.copy_(torch.randn_like(parameter) * 0.02)
    vision_only = score()
    state = get_peft_model_state_dict(model)
    visual_count = sum(".visual." in name for name in state)
    assert visual_count > 0

    def save_adapter(name):
        directory = args.output_dir / name
        directory.mkdir(exist_ok=True)
        model.save_pretrained(directory)
        return str(directory.resolve())

    save_adapter("vision_only")
    for name, parameter in model.named_parameters():
        if ".visual." not in name and (".lora_A." in name or ".lora_B." in name):
            parameter.copy_(torch.randn_like(parameter) * 0.01)
    full = score()
    save_adapter("full")
    # Publish through SkyRL's actual FSDP adapter-export method too. Rank/barrier
    # are replaced because this is an unsharded single-device policy.
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock, patch

    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )
    from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase

    client = Mock(spec=RemoteInferenceClient)
    client.load_lora_adapter = AsyncMock()
    worker = SimpleNamespace(model=SimpleNamespace(model=model), _is_multimodal_lm_only=False)
    sync_directory = args.output_dir / "skyrl_fsdp_full"
    with patch("torch.distributed.get_rank", return_value=0), patch("torch.distributed.barrier"):
        asyncio.run(
            FSDPPolicyWorkerBase._save_lora_adapters_and_sync(
                worker, model, str(sync_directory), client, "measured-policy"
            )
        )
    client.load_lora_adapter.assert_awaited_once()
    from safetensors import safe_open

    exported_file = sync_directory / "adapter_model.safetensors"
    with safe_open(exported_file, framework="pt", device="cpu") as exported:
        exported_keys = sorted(exported.keys())
    exported_visual_keys = [key for key in exported_keys if ".visual." in key]
    assert len(exported_visual_keys) == visual_count and len(exported_keys) == len(state)
    write_json(
        args.output_dir / "fsdp_export.json",
        {
            "export_method": "FSDPPolicyWorkerBase._save_lora_adapters_and_sync",
            "scope": "Actual export on unsharded HF/PEFT policy; rank/barrier and RemoteInferenceClient mocked.",
            "adapter_tensor_count": len(exported_keys),
            "visual_adapter_tensor_count": len(exported_visual_keys),
            "visual_adapter_key_examples": exported_visual_keys[:8],
            "adapter_file_sha256": hashlib.sha256(exported_file.read_bytes()).hexdigest(),
        },
    )
    visual_effect = max(abs(a - b) for a, b in zip(base, vision_only))
    assert visual_effect > 1e-4, f"Visual update too small to diagnose: {visual_effect}"
    data = {
        "model": args.model,
        "model_revision": model.config._commit_hash,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "peft", "transformers", "vllm")},
        "gpu": torch.cuda.get_device_name(),
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "answer_ids": answer_ids,
        "continuation": processor.tokenizer.decode(answer_ids),
        "full_ids": full_ids,
        "image_sha256": hashlib.sha256((args.output_dir / "prompt.png").read_bytes()).hexdigest(),
        "pixel_values_sha256": hashlib.sha256(inputs["pixel_values"].cpu().float().numpy().tobytes()).hexdigest(),
        "image_grid_thw": inputs["image_grid_thw"].cpu().tolist(),
        "visual_adapter_tensor_count": visual_count,
        "adapter_tensor_count": len(state),
        "hf_base": base,
        "hf_vision_only": vision_only,
        "hf_full": full,
        "visual_effect_max_abs": visual_effect,
        "seed": 7819,
        "lora_rank": 8,
        "lora_alpha": 8,
        "visual_factor_std": 0.02,
        "language_factor_std": 0.01,
    }
    write_json(args.output_dir / "policy.json", data)
    print(json.dumps(data, indent=2), flush=True)


def rollout(args):
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor

    policy = json.loads((args.output_dir / "policy.json").read_text())
    image = Image.open(args.output_dir / "prompt.png").convert("RGB")
    processor = AutoProcessor.from_pretrained(
        args.model, revision=args.revision, min_pixels=128 * 128, max_pixels=128 * 128
    )
    reference = processor(text=[policy["prompt"]], images=[image], return_tensors="pt")
    reference_pixels = reference["pixel_values"].cpu()
    assert hashlib.sha256(reference_pixels.float().numpy().tobytes()).hexdigest() == policy["pixel_values_sha256"]
    reference_bf16_hash = hashlib.sha256(reference_pixels.to(torch.bfloat16).float().numpy().tobytes()).hexdigest()
    processed_images = []
    original_process = Qwen3VLMultiModalProcessor._apply_hf_processor_main

    def capture_pixels(self, *process_args, **process_kwargs):
        data = original_process(self, *process_args, **process_kwargs)
        if "pixel_values" in data:
            original_hash = hashlib.sha256(data["pixel_values"].cpu().float().numpy().tobytes()).hexdigest()
            original_shape = list(data["pixel_values"].shape)
            original_difference = data["pixel_values"].cpu().float() - reference_pixels.float()
            pixel_difference = {
                "max_abs": original_difference.abs().max().item(),
                "mean_abs": original_difference.abs().mean().item(),
                "fraction_unequal": (original_difference != 0).float().mean().item(),
            }
            processed_images.append(
                {
                    "original_pixel_values_sha256": original_hash,
                    "original_difference_from_hf": pixel_difference,
                    "pixel_values_shape": original_shape,
                    "pixel_values_sha256": hashlib.sha256(
                        data["pixel_values"].cpu().float().numpy().tobytes()
                    ).hexdigest(),
                    "pixel_values_dtype": str(data["pixel_values"].dtype),
                    "bf16_pixels_sha256": hashlib.sha256(
                        data["pixel_values"].cpu().bfloat16().float().numpy().tobytes()
                    ).hexdigest(),
                    "image_grid_thw": data["image_grid_thw"].cpu().tolist(),
                }
            )
        return data

    Qwen3VLMultiModalProcessor._apply_hf_processor_main = capture_pixels
    llm = LLM(
        model=args.model,
        revision=args.revision,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=8,
        max_loras=1,
        enable_tower_connector_lora=args.stage == "on",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        enforce_eager=True,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        # vLLM 0.33 otherwise returns uint8 CPU pixels and normalizes on GPU.
        # Normalize on CPU here to verify identical tensors before execution.
        mm_device_do_normalize=False,
        limit_mm_per_prompt={"image": 1, "video": 0},
        mm_processor_kwargs={"min_pixels": 128 * 128, "max_pixels": 128 * 128},
        compilation_config={"mode": 0},
        seed=7819,
    )
    # vLLM expects unexpanded image placeholders and expands them itself.
    # Both vision patch embeddings cast pixels to the BF16 model dtype.
    raw_ids = processor.tokenizer.encode(policy["prompt"], add_special_tokens=False) + policy["answer_ids"]
    request = {"prompt_token_ids": raw_ids, "multi_modal_data": {"image": image}}
    result = {
        "stage": args.stage,
        "enable_tower_connector_lora": args.stage == "on",
        "cases": {},
        "hf_bf16_pixels_sha256": reference_bf16_hash,
        "mm_device_do_normalize": False,
        "model_revision": args.revision,
    }
    for adapter_id, (name, directory) in enumerate(
        [("base", None), ("vision_only", "vision_only"), ("full", "skyrl_fsdp_full")], start=1
    ):
        try:
            adapter = LoRARequest(name, adapter_id, str((args.output_dir / directory).resolve())) if directory else None
            output = llm.generate(
                [request],
                SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1),
                lora_request=adapter,
                use_tqdm=False,
            )[0]
            result["processed_images"] = processed_images
            result["returned_prompt_ids"] = output.prompt_token_ids
            # Preserve scores as diagnostics even if an alignment assertion fails.
            values = [
                output.prompt_logprobs[pos][token].logprob
                for pos, token in enumerate(output.prompt_token_ids)
                if pos >= len(output.prompt_token_ids) - len(policy["answer_ids"])
            ]
            result["cases"][name] = {"logprobs": values, "adapter_accepted": True, "alignment_verified": False}
            assert (
                output.prompt_token_ids == policy["full_ids"]
            ), "Expanded vLLM and HF token IDs differ; inspect returned_prompt_ids"
            assert processed_images, "Did not capture vLLM image preprocessing"
            assert processed_images[-1]["bf16_pixels_sha256"] == reference_bf16_hash, "HF/vLLM BF16 input pixels differ"
            assert processed_images[-1]["image_grid_thw"] == policy["image_grid_thw"], "HF/vLLM image grids differ"
            result["cases"][name]["alignment_verified"] = True
        except Exception:
            result["cases"].setdefault(name, {"adapter_accepted": False})["error"] = traceback.format_exc()
        write_json(args.output_dir / f"{args.stage}.json", result)
    print(json.dumps(result, indent=2), flush=True)


def summarize(args):
    policy = json.loads((args.output_dir / "policy.json").read_text())
    rows = []
    direct_effects = []
    stages = {}
    for stage in ("off", "on"):
        path = args.output_dir / f"{stage}.json"
        if not path.exists():
            rows.append(
                {"flag": stage, "error": f"No stage output; inspect {stage}.log for engine initialization failure"}
            )
            continue
        result = json.loads(path.read_text())
        stages[stage] = result
        for case, data in result["cases"].items():
            if not data["adapter_accepted"] or not data.get("alignment_verified") or "error" in data:
                rows.append({"flag": stage, "case": case, "error": data.get("error", "Alignment not verified")})
                continue
            reference = policy["hf_" + case]
            diffs = [abs(a - b) for a, b in zip(reference, data["logprobs"], strict=True)]
            rows.append(
                {
                    "flag": stage,
                    "case": case,
                    "max_abs_logprob_diff": max(diffs),
                    "mean_abs_logprob_diff": sum(diffs) / len(diffs),
                    "tokens": len(diffs),
                }
            )
        base = result["cases"].get("base", {})
        vision = result["cases"].get("vision_only", {})
        if (
            base.get("alignment_verified")
            and vision.get("alignment_verified")
            and "error" not in base
            and "error" not in vision
        ):
            effects = [v - b for v, b in zip(vision["logprobs"], base["logprobs"], strict=True)]
            hf_effects = [v - b for v, b in zip(policy["hf_vision_only"], policy["hf_base"], strict=True)]
            direct_effects.append(
                {
                    "flag": stage,
                    "vllm_vision_only_minus_base": effects,
                    "max_abs_visual_effect": max(map(abs, effects)),
                    "max_abs_effect_difference_from_hf": max(
                        abs(a - b) for a, b in zip(effects, hf_effects, strict=True)
                    ),
                }
            )
    complete = len(rows) == 6 and all("error" not in row for row in rows)
    flag_effects = []
    if complete:
        for case in ("base", "vision_only", "full"):
            off = stages["off"]["cases"][case]["logprobs"]
            on = stages["on"]["cases"][case]["logprobs"]
            flag_effects.append(
                {"case": case, "max_abs_off_vs_on": max(abs(a - b) for a, b in zip(off, on, strict=True))}
            )
    write_json(
        args.output_dir / "summary.json",
        {
            "model": args.model,
            "complete": complete,
            "comparisons": rows,
            "direct_visual_effects": direct_effects,
            "flag_effects": flag_effects,
        },
    )
    print(json.dumps(rows, indent=2), flush=True)
    print(json.dumps(direct_effects, indent=2), flush=True)
    if not complete:
        raise RuntimeError("Comparison incomplete; inspect stage logs and summary.json")


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "policy":
        prepare_policy(args)
    elif args.stage in ("off", "on"):
        (args.output_dir / f"{args.stage}.json").unlink(missing_ok=True)
        rollout(args)
    elif args.stage == "summarize":
        summarize(args)
    else:
        for stage in ("policy", "off", "on"):
            with (args.output_dir / f"{stage}.log").open("w") as log:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--stage",
                    stage,
                    "--model",
                    args.model,
                    "--revision",
                    args.revision,
                    "--output-dir",
                    str(args.output_dir),
                ]
                completed = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            print(f"{stage}: exit {completed.returncode}; log={args.output_dir / (stage + '.log')}", flush=True)
            if stage == "policy" and completed.returncode:
                print((args.output_dir / "policy.log").read_text()[-16000:], flush=True)
                raise RuntimeError("Policy preparation failed; inspect policy.log")
        summarize(args)


if __name__ == "__main__":
    main()
