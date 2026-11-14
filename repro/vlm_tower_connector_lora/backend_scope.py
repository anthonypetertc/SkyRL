"""Execute pinned Megatron matching/name-conversion code without CUDA or torch.

This is a source-level check, not a distributed Megatron training run. Functions
are compiled verbatim from the pinned sources using AST; the mapping registry is
a simple lookup over the real bridge's declarative mappings. It answers whether
default targets include visual modules and how their adapters are named on export.
Only Python's standard library is required. Upstream sources are fetched by SHA.
"""

import argparse
import ast
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from string import digits
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_REV = "7171584bd7452caf2adfa6afd12fb89cc37fbc64"


def function(tree, name, namespace):
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    code = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), "<pinned-source>", "exec"), namespace)
    return namespace[name]


def assignment(tree, name):
    return next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("backend_scope.json"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    assert BRIDGE_REV in (repo / "pyproject.toml").read_text(), "Update the source check when the Bridge pin changes"
    provenance = {}

    def fetch(relative):
        url = (
            f"https://raw.githubusercontent.com/NVIDIA-NeMo/Megatron-Bridge/{BRIDGE_REV}/src/megatron/bridge/{relative}"
        )
        with urllib.request.urlopen(url, timeout=30) as response:
            source = response.read()
        provenance[relative] = {"url": url, "sha256": hashlib.sha256(source).hexdigest()}
        return ast.parse(source)

    targets_tree = ast.parse((repo / "skyrl/backends/skyrl_train/utils/lora_targets.py").read_text())
    namespace = {
        "MEGATRON_ALL_LINEAR_TARGETS": ast.literal_eval(assignment(targets_tree, "MEGATRON_ALL_LINEAR_TARGETS"))
    }
    targets = function(targets_tree, "megatron_all_linear_target_modules", namespace)("lora", True, True, True)
    namespace["re"] = re
    function(fetch("peft/utils.py"), "wildcard_match", namespace)
    matcher = function(fetch("peft/module_matcher.py"), "match", namespace)
    peft_tree = fetch("models/conversion/peft_bridge.py")
    namespace["digits"] = digits
    namespace["MEGATRON_TO_HF_LORA_SUFFIX"] = ast.literal_eval(assignment(peft_tree, "MEGATRON_TO_HF_LORA_SUFFIX"))
    function(peft_tree, "_select_hf_base_param_name", namespace)
    resolve_name = function(peft_tree, "_resolve_hf_adapter_param_name", namespace)
    bridge = fetch("models/qwen_vl/qwen3_vl_bridge.py")
    provider_tree = fetch("models/qwen_vl/qwen3_vl_provider.py")
    dense_provider = next(
        node for node in provider_tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen3VLModelProvider"
    )
    vision_freeze_defaults = {
        node.target.id: ast.literal_eval(node.value)
        for node in dense_provider.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in {"freeze_vision_model", "freeze_vision_projection"}
    }
    assert vision_freeze_defaults == {"freeze_vision_model": False, "freeze_vision_projection": False}
    mappings = ast.literal_eval(assignment(bridge, "param_mappings"))
    # QKV and patch/embed mappings are expressed as constructor calls.
    for node in ast.walk(bridge):
        if not isinstance(node, ast.Call):
            continue
        fields = {kw.arg: kw.value for kw in node.keywords}
        if {"megatron_param", "hf_param"} <= fields.keys():
            try:
                mappings[ast.literal_eval(fields["megatron_param"])] = ast.literal_eval(fields["hf_param"])
            except (ValueError, TypeError):
                pass
    mappings = {name.replace("*", "0"): hf.replace("*", "0") for name, hf in mappings.items()}
    registry = SimpleNamespace(megatron_to_hf_lookup=lambda name: SimpleNamespace(hf_param=mappings[name]))
    matches = []
    owner = SimpleNamespace(
        canonical_mapping={}, target_modules=targets, exclude_modules=[], _record_match=lambda *args: None
    )
    for name in sorted(mappings):
        if not name.startswith("vision_model.") or not name.endswith(".weight"):
            continue
        module = name.removesuffix(".weight")
        parent, _, leaf = module.rpartition(".")
        if matcher(owner, object(), leaf, parent):
            exported = [
                resolve_name(None, registry, module, suffix, ".weight", None)
                for suffix in (".linear_in.weight", ".linear_out.weight")
            ]
            assert all(".visual." in key and ".lora_" in key for key in exported)
            matches.append({"megatron_module": module, "hf_adapter_keys": exported})
    assert any("vision_model.merger." in row["megatron_module"] for row in matches)
    assert any("vision_model.decoder.layers." in row["megatron_module"] for row in matches)
    qwen25_tree = fetch("models/qwen_vl/modeling_qwen25_vl.py")
    assert "Qwen2_5_VisionTransformerPretrainedModel" in ast.unparse(qwen25_tree)
    qwen25_names = [
        "visual.blocks.0.attn.qkv",
        "visual.blocks.0.attn.proj",
        "visual.blocks.0.mlp.gate_proj",
        "visual.blocks.0.mlp.up_proj",
        "visual.blocks.0.mlp.down_proj",
        "visual.merger.mlp.0",
        "visual.merger.mlp.2",
    ]
    qwen25_matches = [
        name for name in qwen25_names if matcher(owner, object(), name.rsplit(".", 1)[1], name.rsplit(".", 1)[0])
    ]
    assert not qwen25_matches
    result = {
        "bridge_revision": BRIDGE_REV,
        "default_megatron_lora_targets": targets,
        "dense_qwen3_vl_vision_freeze_defaults": vision_freeze_defaults,
        "qwen3_vl_visual_matches_and_export_names": matches,
        "qwen25_vl_representative_visual_names": qwen25_names,
        "qwen25_vl_default_visual_matches": qwen25_matches,
        "provenance": provenance,
        "scope": "Verbatim target matcher and adapter name resolver on pinned bridge mappings; no model, tensors, collectives, or adapter transport.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
