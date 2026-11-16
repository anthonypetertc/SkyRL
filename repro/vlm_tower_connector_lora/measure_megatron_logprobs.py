"""Measure an actual SkyRL Megatron VLM policy against vLLM tower LoRA off/on.

One real process-group rank, native Megatron model/forward and native Bridge
adapter export. Primary scores use Megatron; HF supplies checkpoint files and
input processing. Optional hf-export scores unchanged exported tensors in HF
with exact-path target selection as a separate diagnostic.
No optimizer is constructed; adapters receive seeded nonzero factors.
"""

import argparse
import asyncio
import ctypes
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

from measure_logprobs import rollout, summarize, write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "policy", "off", "on", "summarize", "hf-export"], default="all")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--revision", default="89644892e4d85e24eaac8bacfd4f463576704203")
    parser.add_argument("--output-dir", type=Path, default=Path("vlm_megatron_measurement"))
    return parser.parse_args()


def ensure_cuda13_nvrtc(args):
    """Select the compiler shipped with this CUDA-13 environment before TE loads.

    The Platform base image also supplies CUDA 12.6. TE otherwise finds that
    compiler while using the installed CUDA-13 headers, which fails compilation.
    This changes environment library selection, not model math or adapter data.
    """
    if os.environ.get("SKYRL_MEASUREMENT_NVRTC_READY"):
        return
    site = Path(sysconfig.get_paths()["purelib"])
    candidates = sorted((site / "nvidia").glob("**/libnvrtc.so.13*"), key=lambda p: len(str(p)))
    assert candidates, f"CUDA-13 NVRTC library missing under {site}"
    library = candidates[0].resolve()
    builtins = list(library.parent.glob("libnvrtc-builtins.so.13*"))
    assert builtins, f"Matching NVRTC builtins missing next to {library}"
    compiler = ctypes.CDLL(str(library))
    major, minor = ctypes.c_int(), ctypes.c_int()
    assert compiler.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)) == 0
    assert major.value == 13, (major.value, minor.value)
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(library.parent) + ":" + environment.get("LD_LIBRARY_PATH", "")
    environment["LD_PRELOAD"] = str(library) + ":" + environment.get("LD_PRELOAD", "")
    include = library.parent.parent / "include"
    assert (include / "cuda_runtime.h").exists(), f"CUDA-13 headers missing at {include}"
    header = (include / "cuda.h").read_text()
    assert "#define CUDA_VERSION 130" in header, "Expected CUDA-13 headers"
    environment["NVTE_CUDA_INCLUDE_DIR"] = str(include)
    environment["SKYRL_MEASUREMENT_NVRTC_READY"] = "1"
    write_json(
        args.output_dir
        / ("hf_export_runtime_libraries.json" if args.stage == "hf-export" else "runtime_libraries.json"),
        {
            "nvrtc_library": str(library),
            "nvrtc_version": [major.value, minor.value],
            "nvrtc_builtins": [str(path) for path in builtins],
            "cuda_include": str(include),
            "reason": "Avoid base-image CUDA-12.6 NVRTC with pinned CUDA-13 TE and headers",
        },
    )
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)


def prepare_policy(args):
    import megatron.core.parallel_state as mpu
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image, ImageDraw
    from safetensors import safe_open
    from transformers import AutoProcessor

    from skyrl.backends.skyrl_train.training_batch import TensorList, TrainingInputBatch
    from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
        MegatronPolicyWorkerBase,
    )
    from skyrl.train.config import SkyRLLoraConfig, SkyRLTrainConfig
    from skyrl.train.utils.utils import validate_cfg

    checkpoint = snapshot_download(args.model, revision=args.revision)
    processor = AutoProcessor.from_pretrained(checkpoint, min_pixels=128 * 128, max_pixels=128 * 128)
    image = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 16, 60, 112), fill="red")
    draw.ellipse((68, 32, 120, 96), fill="blue")
    image.save(args.output_dir / "prompt.png")
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
    answer_ids = processor.tokenizer.encode(
        "There is a red rectangle and a blue circle on a white background.", add_special_tokens=False
    )[:12]
    prompt_ids = inputs["input_ids"][0].tolist()
    full_ids = prompt_ids + answer_ids
    sequences = torch.tensor([full_ids])
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": torch.ones_like(sequences),
            "loss_mask": torch.ones(1, len(answer_ids)),
            "response_mask": torch.ones(1, len(answer_ids)),
            "pixel_values": TensorList([inputs["pixel_values"]]),
            "image_grid_thw": TensorList([inputs["image_grid_thw"]]),
        }
    )
    batch.metadata = {"response_length": len(answer_ids)}

    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.model.path = checkpoint
    cfg.trainer.policy.model.lora = SkyRLLoraConfig(
        rank=8,
        alpha=8,
        dropout=0.0,
        target_modules="all-linear",
        lora_sync_path=str(args.output_dir / "skyrl_megatron_full"),
        sync_mode="disk",
    )
    cfg.trainer.policy.language_model_only = False
    cfg.trainer.policy.inference_only_init = True
    cfg.trainer.bf16 = True
    cfg.trainer.flash_attn = False
    cfg.trainer.gradient_checkpointing = False
    cfg.trainer.use_expandable_segments = False
    cfg.trainer.remove_microbatch_padding = False
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.algorithm.temperature = 1.0
    cfg.trainer.logger = "console"
    cfg.trainer.seed = 7819
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    mc = cfg.trainer.policy.megatron_config
    mc.tensor_model_parallel_size = 1
    mc.pipeline_model_parallel_size = 1
    mc.context_parallel_size = 1
    mc.expert_model_parallel_size = 1
    mc.expert_tensor_parallel_size = 1
    mc.lora_config.merge_lora = False
    validate_cfg(cfg)

    worker = MegatronPolicyWorkerBase(
        cfg=cfg.trainer,
        world_size=1,
        rank=0,
        local_rank=0,
        master_addr="127.0.0.1",
        master_port=29681,
        sequence_parallel_size=1,
    )
    worker.init_worker_process_group()
    worker.init_model(checkpoint)
    assert worker.is_vlm and worker.optimizer is None
    parameters = {
        f"chunk{i}.{name}": parameter
        for i, chunk in enumerate(worker.actor_module)
        for name, parameter in chunk.named_parameters()
    }
    adapters = {
        name: parameter
        for name, parameter in parameters.items()
        if ".adapter." in name and name.endswith(("linear_in.weight", "linear_out.weight"))
    }
    visual = {name: p for name, p in adapters.items() if "vision_model." in name or "vision_projection." in name}
    language = {name: p for name, p in adapters.items() if name not in visual}
    inventory = [
        {
            "name": name,
            "shape": list(p.shape),
            "dtype": str(p.dtype),
            "requires_grad": p.requires_grad,
            "visual": name in visual,
        }
        for name, p in adapters.items()
    ]
    write_json(
        args.output_dir / "megatron_parameters.json",
        {
            "adapters": inventory,
            "all_trainable_names": [name for name, p in parameters.items() if p.requires_grad],
            "module_classes": sorted(
                {
                    type(module).__module__ + "." + type(module).__qualname__
                    for chunk in worker.actor_module
                    for _, module in chunk.named_modules()
                }
            ),
        },
    )
    assert visual and language, "No actual visual/language adapters found; inspect parameter inventory"
    assert all(p.requires_grad for p in adapters.values())
    assert all(torch.count_nonzero(p).item() == 0 for n, p in adapters.items() if n.endswith("linear_out.weight"))

    def score():
        result = worker.forward(batch)
        values = result.loss_fn_outputs[0]["logprobs"]
        assert len(values) == len(answer_ids), (len(values), len(answer_ids))
        assert torch.isfinite(torch.tensor(values)).all()
        return values

    def perturb(selected, std):
        with torch.no_grad():
            for name, parameter in sorted(selected.items()):
                seed = 7819 + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
                generator = torch.Generator(device="cpu").manual_seed(seed)
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * std)

    exports = {}

    class ExportSink:
        """Records publication only; vLLM subsequently loads these exact files."""

        async def update_named_weights(self, request):
            self.request = request

    def export(case, directory):
        sink = ExportSink()
        path = args.output_dir / directory
        asyncio.run(worker._save_lora_adapters_and_sync(str(path), sink, lora_name=case))
        assert Path(sink.request.lora_path) == path
        weights = path / "adapter_model.safetensors"
        with safe_open(weights, framework="pt", device="cpu") as handle:
            keys = sorted(handle.keys())
            visual_keys = [key for key in keys if ".visual." in key]
            tensors = [
                {
                    "name": key,
                    "shape": list(handle.get_tensor(key).shape),
                    "dtype": str(handle.get_tensor(key).dtype),
                    "nonzero": int(torch.count_nonzero(handle.get_tensor(key))),
                }
                for key in keys
            ]
        assert visual_keys, "Real runtime export has no visual adapter keys"
        assert all(tensor["nonzero"] > 0 for tensor in tensors if tensor["name"] in visual_keys)
        language_b = [tensor for tensor in tensors if ".visual." not in tensor["name"] and ".lora_B." in tensor["name"]]
        if case == "vision_only":
            assert language_b and all(tensor["nonzero"] == 0 for tensor in language_b)
        exports[case] = {
            "method": "MegatronPolicyWorkerBase._save_lora_adapters_and_sync",
            "adapter_tensor_count": len(keys),
            "visual_adapter_tensor_count": len(visual_keys),
            "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "tensors": tensors,
            "config": json.loads((path / "adapter_config.json").read_text()),
        }
        write_json(args.output_dir / "megatron_export.json", exports)

    base = score()
    perturb(visual, 0.02)
    assert all(torch.count_nonzero(p).item() == 0 for n, p in language.items() if n.endswith("linear_out.weight"))
    vision_only = score()
    visual_effect = max(abs(a - b) for a, b in zip(base, vision_only, strict=True))
    assert visual_effect > 1e-4, f"Megatron visual update has negligible effect: {visual_effect}"
    export("vision_only", "skyrl_megatron_vision_only")
    perturb(language, 0.01)
    full = score()
    export("full", "skyrl_megatron_full")
    versions = {}
    for name in ("torch", "transformers", "peft", "vllm", "megatron-core", "megatron-bridge", "transformer-engine"):
        distribution = importlib.metadata.distribution(name)
        versions[name] = {
            "version": distribution.version,
            "direct_url": json.loads(distribution.read_text("direct_url.json") or "null"),
        }
    data = {
        "model": args.model,
        "model_revision": args.revision,
        "checkpoint_snapshot": Path(checkpoint).name,
        "reference_backend": "megatron",
        "loaded_nvrtc_libraries": sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "nvrtc" in line and ".so" in line
            }
        ),
        "policy_class": type(worker).__module__ + "." + type(worker).__name__,
        "provider_class": type(worker.provider).__module__ + "." + type(worker.provider).__name__,
        "versions": versions,
        "gpu": torch.cuda.get_device_name(),
        "parallelism": {
            "tp": mpu.get_tensor_model_parallel_world_size(),
            "pp": mpu.get_pipeline_model_parallel_world_size(),
            "cp": mpu.get_context_parallel_world_size(),
        },
        "merge_lora": False,
        "optimizer_constructed": False,
        "transport": "publication sink; direct disk load",
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "answer_ids": answer_ids,
        "full_ids": full_ids,
        "continuation": processor.tokenizer.decode(answer_ids),
        "pixel_values_sha256": hashlib.sha256(inputs["pixel_values"].float().numpy().tobytes()).hexdigest(),
        "image_grid_thw": inputs["image_grid_thw"].tolist(),
        "image_sha256": hashlib.sha256((args.output_dir / "prompt.png").read_bytes()).hexdigest(),
        "megatron_base": base,
        "megatron_vision_only": vision_only,
        "megatron_full": full,
        "visual_effect_max_abs": visual_effect,
        "visual_adapter_parameter_count": len(visual),
        "adapter_parameter_count": len(adapters),
        "seed": 7819,
        "visual_factor_std": 0.02,
        "language_factor_std": 0.01,
        "lora_rank": 8,
        "lora_alpha": 8,
        "adapter_directories": [
            ["base", None],
            ["vision_only", "skyrl_megatron_vision_only"],
            ["full", "skyrl_megatron_full"],
        ],
    }
    write_json(args.output_dir / "policy.json", data)
    print(json.dumps(data, indent=2), flush=True)
    torch.distributed.destroy_process_group()


def diagnose_hf_export(args):
    """Score the unchanged Megatron artifacts in HF; never regenerate adapters."""
    import torch
    from peft import (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    policy = json.loads((args.output_dir / "policy.json").read_text())
    processor = AutoProcessor.from_pretrained(
        args.model, revision=args.revision, min_pixels=128 * 128, max_pixels=128 * 128
    )
    image = Image.open(args.output_dir / "prompt.png").convert("RGB")
    inputs = processor(text=[policy["prompt"]], images=[image], return_tensors="pt")
    assert inputs["input_ids"][0].tolist() == policy["prompt_ids"]
    assert inputs["image_grid_thw"].tolist() == policy["image_grid_thw"]
    assert hashlib.sha256(inputs["pixel_values"].float().numpy().tobytes()).hexdigest() == policy["pixel_values_sha256"]
    inputs["input_ids"] = torch.tensor([policy["full_ids"]])
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    for key in ("mm_token_type_ids", "token_type_ids"):
        if key in inputs:
            inputs[key] = torch.cat(
                [inputs[key], torch.zeros((1, len(policy["answer_ids"])), dtype=inputs[key].dtype)], dim=1
            )
    inputs = {key: value.cuda() for key, value in inputs.items()}
    result = {
        "model": args.model,
        "revision": args.revision,
        "adapter_dtype": "bfloat16",
        "attention": "eager",
        "cases": {},
    }
    for case, directory in policy["adapter_directories"]:
        model = (
            AutoModelForImageTextToText.from_pretrained(
                args.model, revision=args.revision, dtype=torch.bfloat16, attn_implementation="eager"
            )
            .cuda()
            .eval()
        )
        load_info = None
        if directory:
            path = args.output_dir / directory
            weights = path / "adapter_model.safetensors"
            before = hashlib.sha256(weights.read_bytes()).hexdigest()
            state = load_file(weights)
            config = LoraConfig.from_pretrained(path)
            original_targets = config.target_modules
            # Bridge's suffix targets also match HF patch_embed.proj, which has
            # no native exported adapter. Wrap only modules in the actual file.
            config.target_modules = {key.removeprefix("base_model.model.").split(".lora_")[0] for key in state}
            model = get_peft_model(model, config, autocast_adapter_dtype=False).eval()
            load_result = set_peft_model_state_dict(model, state)
            missing = [key for key in load_result.missing_keys if ".lora_" in key]
            assert not missing and not load_result.unexpected_keys, (missing, load_result.unexpected_keys)
            restored = get_peft_model_state_dict(model)
            assert set(restored) == set(state), (set(restored) - set(state), set(state) - set(restored))
            assert all(torch.equal(restored[key].cpu(), value) for key, value in state.items())
            assert all(restored[key].dtype == value.dtype for key, value in state.items())
            assert before == hashlib.sha256(weights.read_bytes()).hexdigest()
            load_info = {
                "sha256": before,
                "tensor_count": len(state),
                "missing_adapter_keys": missing,
                "unexpected_keys": load_result.unexpected_keys,
                "tensors_and_dtypes_exact": True,
                "original_target_modules": sorted(original_targets),
                "target_selection": "Exact module paths from artifact keys; no tensor changes",
            }
        with torch.no_grad():
            logits = model(**inputs, use_cache=False).logits[0]
            positions = torch.arange(len(policy["prompt_ids"]) - 1, len(policy["full_ids"]) - 1, device="cuda")
            target = torch.tensor(policy["answer_ids"], device="cuda")
            values = logits[positions].float().log_softmax(-1).gather(1, target[:, None]).flatten().cpu().tolist()
        assert torch.isfinite(torch.tensor(values)).all()
        references = {"megatron": policy["megatron_" + case]}
        for flag in ("off", "on"):
            rollout_data = json.loads((args.output_dir / f"{flag}.json").read_text())
            references["vllm_" + flag] = rollout_data["cases"][case]["logprobs"]
        comparisons = {}
        for name, reference in references.items():
            differences = [abs(a - b) for a, b in zip(values, reference, strict=True)]
            comparisons[name] = {"max_abs": max(differences), "mean_abs": sum(differences) / len(differences)}
        result["cases"][case] = {"hf_logprobs": values, "adapter_load": load_info, "comparisons": comparisons}
        write_json(args.output_dir / "hf_export_diagnostic.json", result)
        del model, logits
        torch.cuda.empty_cache()
    result["complete"] = True
    write_json(args.output_dir / "hf_export_diagnostic.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage != "summarize":
        ensure_cuda13_nvrtc(args)
    if args.stage == "policy":
        prepare_policy(args)
    elif args.stage == "hf-export":
        (args.output_dir / "hf_export_diagnostic.json").unlink(missing_ok=True)
        diagnose_hf_export(args)
    elif args.stage in ("off", "on"):
        (args.output_dir / f"{args.stage}.json").unlink(missing_ok=True)
        rollout(args)
    elif args.stage == "summarize":
        summarize(args)
    else:
        environment = os.environ.copy()
        environment["SKYRL_DUMP_INFRA_LOG_TO_STDOUT"] = "1"
        for stage in ("policy", "off", "on"):
            with (args.output_dir / f"{stage}.log").open("w") as log:
                result = subprocess.run(
                    [
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
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
            print(f"{stage}: exit {result.returncode}", flush=True)
            if result.returncode:
                print((args.output_dir / f"{stage}.log").read_text()[-18000:], flush=True)
                raise RuntimeError(f"{stage} failed; inspect its log")
        summarize(args)


if __name__ == "__main__":
    main()
