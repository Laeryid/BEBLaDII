#!/usr/bin/env python3
"""
Level 4 Validator: TPU HBM Memory Budget & Peak Allocation Estimator
Analyzes AST and hyperparameters of a TPU training script/notebook to calculate:
1. Static Model & Optimizer memory (weights, gradients, AdamW moments).
2. Peak Projection & Vocabulary matrix allocations (HLO Temp / All-Reduce buffers).
3. Activation memory during forward/backward passes.
4. Total HBM peak demand vs target TPU hardware limits (Kaggle TPU 16 GB, GCP v6e 32 GB).
"""

import ast
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple


# Target TPU Specifications (HBM in GB)
TPU_TARGETS = {
    "kaggle_v3": {"name": "Kaggle TPU v3-8", "hbm_total": 16.0, "hbm_usable": 15.75, "chips": 8},
    "kaggle_v5e": {"name": "Kaggle TPU v5e-8", "hbm_total": 16.0, "hbm_usable": 15.75, "chips": 8},
    "gcp_v6e": {"name": "GCP Cloud TPU v6e-4/8", "hbm_total": 32.0, "hbm_usable": 31.20, "chips": 8},
}


class TPUParamExtractor(ast.NodeVisitor):
    def __init__(self):
        self.params: Dict[str, Any] = {
            "batch_size": None,
            "max_length": 512,
            "max_length_a": None,
            "max_length_q": None,
            "hidden_dim": 1024,
            "num_layers": 40,
            "unfreeze_k": 0,
            "dict_size": 151936,  # Default Qwen/Latent dict vocab
            "dict_chunks": 1,
            "dict_dtype": "float32",
            "has_ca_layers": False,
            "has_dus": False,
            "has_qwen": False,
            "has_vae": False,
            "use_gradient_checkpointing": False,
        }

    def _eval_num(self, node: ast.AST) -> Any:
        try:
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                left = self._eval_num(node.left)
                right = self._eval_num(node.right)
                if left is not None and right is not None:
                    if isinstance(node.op, ast.Mult): return left * right
                    elif isinstance(node.op, ast.Add): return left + right
                    elif isinstance(node.op, ast.Sub): return left - right
                    elif isinstance(node.op, ast.Div): return left / right
                    elif isinstance(node.op, ast.FloorDiv): return left // right
        except Exception:
            pass
        return None

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            target_id = None
            if isinstance(target, ast.Name):
                target_id = target.id
            elif isinstance(target, ast.Attribute):
                target_id = target.attr

            if target_id:
                val = self._eval_num(node.value)
                if target_id == "batch_size" and val is not None:
                    self.params["batch_size"] = int(val)
                elif target_id in ("max_length", "max_length_a", "max_length_q") and val is not None:
                    self.params[target_id] = int(val)
                elif target_id == "unfreeze_k_after_ca" and val is not None:
                    self.params["unfreeze_k"] = int(val)
                elif target_id == "use_gradient_checkpointing":
                    if isinstance(node.value, ast.Constant):
                        self.params["use_gradient_checkpointing"] = bool(node.value.value)

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Detect .chunk(N) call on latent_dict / dictionary
        if isinstance(node.func, ast.Attribute) and node.func.attr == "chunk":
            if node.args and len(node.args) >= 1:
                chunk_val = self._eval_num(node.args[0])
                if chunk_val is not None and chunk_val > self.params["dict_chunks"]:
                    self.params["dict_chunks"] = int(chunk_val)

        # Detect register_buffer for latent_dict
        if isinstance(node.func, ast.Attribute) and node.func.attr == "register_buffer":
            code_str = ast.unparse(node)
            if "latent_dict" in code_str:
                if "bfloat16" in code_str or "bf16" in code_str:
                    self.params["dict_dtype"] = "bfloat16"
                elif "float" in code_str or "float32" in code_str:
                    self.params["dict_dtype"] = "float32"

        # Detect model components
        func_str = ast.unparse(node.func)
        if "DUSModel" in func_str or "ModernBERT" in func_str:
            self.params["has_dus"] = True
        if "CAPromptLayer" in func_str:
            self.params["has_ca_layers"] = True
        if "LatentEncoder" in func_str:
            self.params["has_vae"] = True
        if "qwen" in func_str.lower() or "auto_model" in func_str.lower():
            self.params["has_qwen"] = True

        self.generic_visit(node)


def estimate_hbm_usage(params: Dict[str, Any], target_key: str = "kaggle_v3") -> Dict[str, Any]:
    target = TPU_TARGETS.get(target_key, TPU_TARGETS["kaggle_v3"])
    
    batch_size = params["batch_size"] or 64
    seq_len = params["max_length_a"] or params["max_length"] or 512
    chips = target["chips"]
    batch_per_chip = max(1, batch_size // chips)
    hidden_dim = params["hidden_dim"] or 1024

    # 1. Static Model Weights (GB)
    # ModernBERT / DUS (40 layers, 1024 dim) ~ 400M params
    # CA Layers (3 layers: out_proj, qkv, norm) ~ 15M params
    # VAE / Encoder ~ 100M params (frozen BF16)
    # Qwen Embeddings ~ 151936 * 1536 / 1024 ~ 230M params (frozen BF16)
    
    trainable_params_m = 15.0  # CA base
    unfreeze_k = params["unfreeze_k"]
    if unfreeze_k > 0:
        # Each ModernBERT layer ~ 10M params
        trainable_params_m += min(39, unfreeze_k * 3) * 10.0

    frozen_params_m = 400.0 - (trainable_params_m - 15.0) + 100.0 + 230.0

    # Trainable: FP32 weights (4B) + Gradients (4B) + AdamW 1st moment (4B) + 2nd moment (4B) = 16B/param
    mem_trainable_gb = (trainable_params_m * 1e6 * 16.0) / (1024 ** 3)
    # Frozen: BF16 weights (2B)
    mem_frozen_gb = (frozen_params_m * 1e6 * 2.0) / (1024 ** 3)

    # Dictionary buffer (latent_dict)
    dict_size = params["dict_size"]
    dict_dtype_bytes = 2.0 if params["dict_dtype"] == "bfloat16" else 4.0
    mem_dict_buffer_gb = (dict_size * hidden_dim * dict_dtype_bytes) / (1024 ** 3)

    mem_static_total_gb = mem_trainable_gb + mem_frozen_gb + mem_dict_buffer_gb

    # 2. Peak Projection / Vocabulary Chunking Memory (HLO Temp)
    # sims_chunk = z_noisy @ dict_chunk.T
    # Shape: [batch_size, seq_len, dict_size / dict_chunks]
    dict_chunks = max(1, params["dict_chunks"])
    chunk_vocab = dict_size / dict_chunks
    
    # In XLA, both the computation fusion and the all-reduce buffer exist concurrently in HLO temp (2x multiplier)
    mem_single_sims_gb = (batch_size * seq_len * chunk_vocab * dict_dtype_bytes) / (1024 ** 3)
    mem_peak_projection_gb = mem_single_sims_gb * 2.0

    # 3. Activations (Forward & Backward)
    # For transformer layers per chip:
    # ModernBERT layers: batch_per_chip * seq_len * hidden_dim * layers * bytes
    act_bytes = 2.0  # BF16 activations
    num_layers = params["num_layers"]
    # Rough rule of thumb: ~34 * B_local * T * D * L bytes for standard backward activations without GC
    if params["use_gradient_checkpointing"]:
        mem_activations_gb = (batch_per_chip * seq_len * hidden_dim * 4.0 * act_bytes) / (1024 ** 3)
    else:
        mem_activations_gb = (batch_per_chip * seq_len * hidden_dim * num_layers * 4.0 * act_bytes) / (1024 ** 3)

    # 4. XLA Runtime & Compiler Workspace Overhead
    mem_xla_runtime_gb = 1.25

    # Total Peak HBM Estimate
    total_peak_gb = mem_static_total_gb + mem_peak_projection_gb + mem_activations_gb + mem_xla_runtime_gb
    usable_hbm = target["hbm_usable"]

    status = "PASS"
    if total_peak_gb > usable_hbm:
        status = "DANGER"
    elif total_peak_gb > (usable_hbm * 0.85):
        status = "WARNING"

    return {
        "target": target,
        "batch_size": batch_size,
        "batch_per_chip": batch_per_chip,
        "seq_len": seq_len,
        "dict_chunks": dict_chunks,
        "dict_dtype": params["dict_dtype"],
        "mem_static_gb": mem_static_total_gb,
        "mem_trainable_gb": mem_trainable_gb,
        "mem_dict_buffer_gb": mem_dict_buffer_gb,
        "mem_peak_projection_gb": mem_peak_projection_gb,
        "mem_single_sims_gb": mem_single_sims_gb,
        "mem_activations_gb": mem_activations_gb,
        "mem_xla_runtime_gb": mem_xla_runtime_gb,
        "total_peak_gb": total_peak_gb,
        "usable_hbm": usable_hbm,
        "status": status,
    }


def analyze_script(script_path: Path, target_key: str = "kaggle_v3") -> int:
    if not script_path.exists():
        print(f"[FAIL] Target script not found: {script_path}", file=sys.stderr)
        return 1

    try:
        with open(script_path, "r", encoding="utf-8-sig", errors="replace") as f:
            code = f.read()
        tree = ast.parse(code, filename=str(script_path))
    except Exception as e:
        print(f"[FAIL] Failed to parse script AST: {e}", file=sys.stderr)
        return 1

    extractor = TPUParamExtractor()
    extractor.visit(tree)

    res = estimate_hbm_usage(extractor.params, target_key=target_key)
    target = res["target"]

    print(f"\n{'='*65}")
    print(f"=== [Level 4] TPU HBM Memory Budget & Peak Allocation Estimator")
    print(f"{'='*65}")
    print(f"Target Hardware:       {target['name']} ({target['hbm_total']:.1f} GB HBM, usable: {target['hbm_usable']:.2f} GB)")
    print(f"Global Batch Size:     {res['batch_size']} ({res['batch_per_chip']} per TPU chip)")
    print(f"Sequence Length (T):   {res['seq_len']}")
    print(f"Dictionary Chunking:   {res['dict_chunks']} chunk(s) (dtype: {res['dict_dtype']})")
    print(f"{'-'*65}")
    print(f"Estimated Breakdown:")
    print(f"  • Static Model & AdamW Buffers:    {res['mem_static_gb']:6.2f} GB")
    print(f"      (Trainable + AdamW: {res['mem_trainable_gb']:.2f} GB | Dict Buffer: {res['mem_dict_buffer_gb']:.2f} GB)")
    print(f"  • Activations (Fwd/Bwd):            {res['mem_activations_gb']:6.2f} GB")
    print(f"  • Peak Projection Buffer (Sims):    {res['mem_peak_projection_gb']:6.2f} GB  (2x XLA fusion/all-reduce)")
    print(f"  • XLA Runtime Reserve & Workspace:  {res['mem_xla_runtime_gb']:6.2f} GB")
    print(f"{'-'*65}")
    print(f"ESTIMATED PEAK HBM DEMAND:           {res['total_peak_gb']:6.2f} GB / {res['usable_hbm']:.2f} GB")

    diff = res['total_peak_gb'] - res['usable_hbm']
    if res["status"] == "DANGER":
        print(f"\n[DANGER / OOM RISK] Peak memory exceeds usable HBM by {diff:+.2f} GB!")
        print(f"Actionable Fixes:")
        if res['mem_peak_projection_gb'] > 3.0:
            print(f"  -> Critical: Increase dictionary chunks (e.g. .chunk(16) or .chunk(32)).")
            print(f"  -> Critical: Switch latent_dict and matmul to bfloat16.")
        if res['batch_size'] > 64:
            print(f"  -> Reduce global batch_size to 64 or 32.")
        return 1
    elif res["status"] == "WARNING":
        print(f"\n[WARNING] Peak memory is close to HBM limit ({res['total_peak_gb']/res['usable_hbm']*100:.1f}% capacity).")
        print(f"Monitor training closely for compiler fragmentation.")
        return 0
    else:
        print(f"\n[PASS] Memory budget is safe ({res['total_peak_gb']/res['usable_hbm']*100:.1f}% capacity). Script fits comfortably in HBM.")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Estimate TPU HBM memory demand and detect OOM risks.")
    parser.add_argument("script", help="Path to TPU training script to analyze")
    parser.add_argument("--target", default="kaggle_v3", choices=list(TPU_TARGETS.keys()),
                        help="Target TPU architecture (default: kaggle_v3)")
    args = parser.parse_args()

    sys.exit(analyze_script(Path(args.script).resolve(), target_key=args.target))


if __name__ == "__main__":
    main()
