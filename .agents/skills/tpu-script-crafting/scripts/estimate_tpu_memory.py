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
            "min_ca_layer_idx": 11,  # Lowest layer index requiring grad (e.g. CA at layer 12 -> idx 11)
            "dus_dtype": "bfloat16",
            "dict_size": 151936,  # Default Qwen/Latent dict vocab
            "dict_chunks": 1,
            "dict_dtype": "float32",
            "has_ca_layers": False,
            "has_dus": False,
            "has_qwen": False,
            "has_vae": False,
            "use_gradient_checkpointing": False,
            "has_mid_step_loss_sync": False,
            "has_outputs_after_mark_step": False,
            "has_device_ema_backup": False,
            "has_resume_after_fsdp": False,
            "run_diagnosis_first": False,
        }
        self._in_train_loop = False
        self._saw_forward_in_loop = False
        self._saw_backward_in_loop = False
        self._saw_mark_step_in_loop = False
        self._saw_fsdp_in_main = False

    def _eval_num(self, node: ast.AST) -> Any:
        try:
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.IfExp):
                # For expressions like: 2 if os.environ.get("SMOKE_TEST") else 64*2
                # Evaluate both branches and take production (orelse) or max numeric value
                body_val = self._eval_num(node.body)
                else_val = self._eval_num(node.orelse)
                if else_val is not None and body_val is not None:
                    return max(body_val, else_val)
                return else_val if else_val is not None else body_val
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

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name == "apply_shadow":
            code_str = ast.unparse(node) if hasattr(ast, "unparse") else ""
            if ".clone()" in code_str and ".cpu()" not in code_str:
                self.params["has_device_ema_backup"] = True
        if node.name == "main":
            self._saw_fsdp_in_main = False
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        iter_str = ast.unparse(node.iter) if hasattr(ast, "unparse") else ""
        is_dl = "dataloader" in iter_str and "val" not in iter_str
        prev_loop = self._in_train_loop
        if is_dl:
            self._in_train_loop = True
            self._saw_forward_in_loop = False
            self._saw_backward_in_loop = False
            self._saw_mark_step_in_loop = False
        self.generic_visit(node)
        self._in_train_loop = prev_loop

    def visit_If(self, node: ast.If):
        if self._in_train_loop and self._saw_forward_in_loop and not self._saw_backward_in_loop:
            test_str = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
            if any(k in test_str for k in ["isfinite(loss)", "isnan(loss)", "isinf(loss)"]):
                self.params["has_mid_step_loss_sync"] = True
        self.generic_visit(node)

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
                elif target_id == "run_diagnosis_first":
                    if isinstance(node.value, ast.Constant):
                        self.params["run_diagnosis_first"] = bool(node.value.value)
                elif target_id == "outputs" and self._in_train_loop:
                    self._saw_forward_in_loop = True

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func_str = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
        full_str = ast.unparse(node) if hasattr(ast, "unparse") else ""

        # Track backward and mark_step inside train loop
        if self._in_train_loop:
            if func_str.endswith(".backward"):
                self._saw_backward_in_loop = True
            elif func_str in ("xm.mark_step", "torch_xla.sync") and self._saw_backward_in_loop:
                self._saw_mark_step_in_loop = True
            elif self._saw_mark_step_in_loop and "content_cos_stats(outputs)" in full_str:
                self.params["has_outputs_after_mark_step"] = True

        # Track FSDP vs load_state_dict order in main
        if "SpmdFullyShardedDataParallel" in func_str:
            self._saw_fsdp_in_main = True
        if self._saw_fsdp_in_main and func_str.endswith("model.load_state_dict"):
            self.params["has_resume_after_fsdp"] = True

        # Detect DUS precision: self.dus.to(torch.float32)
        if "self.dus.to" in func_str and ("float32" in full_str or "torch.float" in full_str):
            self.params["dus_dtype"] = "float32"

        # Detect .chunk(N) call on latent_dict / dictionary
        if isinstance(node.func, ast.Attribute) and node.func.attr == "chunk":
            if node.args and len(node.args) >= 1:
                chunk_val = self._eval_num(node.args[0])
                if chunk_val is not None and chunk_val > self.params["dict_chunks"]:
                    self.params["dict_chunks"] = int(chunk_val)

        # Detect register_buffer for latent_dict
        if isinstance(node.func, ast.Attribute) and node.func.attr == "register_buffer":
            if "latent_dict" in full_str:
                if "bfloat16" in full_str or "bf16" in full_str:
                    self.params["dict_dtype"] = "bfloat16"
                elif "float" in full_str or "float32" in full_str:
                    self.params["dict_dtype"] = "float32"

        # Detect model components
        if "DUSModel" in func_str or "ModernBERT" in func_str:
            self.params["has_dus"] = True
        if "CAPromptLayer" in func_str:
            self.params["has_ca_layers"] = True
        if "LatentEncoder" in func_str:
            self.params["has_vae"] = True
        if "qwen" in func_str.lower() or "auto_model" in func_str.lower() or "automodel" in func_str.lower():
            self.params["has_qwen"] = True

        self.generic_visit(node)


def estimate_hbm_usage(params: Dict[str, Any], target_key: str = "kaggle_v3") -> Dict[str, Any]:
    target = TPU_TARGETS.get(target_key, TPU_TARGETS["kaggle_v3"])
    
    batch_size = params["batch_size"] or 64
    seq_len = params["max_length_a"] or params["max_length"] or 512
    chips = target["chips"]
    batch_per_chip = max(1, batch_size // chips)
    hidden_dim = params["hidden_dim"] or 1024

    # 1. Static Model Weights + EMA Shadow (GB)
    # ModernBERT / DUS (40 layers, 1024 dim) ~ 500M params (12.6M per layer)
    # CA Layers (3 layers: out_proj, qkv, norm, adaln) ~ 25.2M params
    # VAE / Encoder ~ 100M params (frozen BF16)
    # Qwen Embeddings ~ 151936 * 1536 ~ 233M params (frozen BF16)
    
    ca_params_m = 25.2 if params["has_ca_layers"] else 0.0
    unfreeze_k = params["unfreeze_k"]
    unfrozen_dus_layers = min(39, unfreeze_k * 3) if unfreeze_k > 0 else 0
    unfrozen_dus_params_m = unfrozen_dus_layers * 12.6
    trainable_params_m = ca_params_m + unfrozen_dus_params_m

    frozen_dus_layers = 40 - unfrozen_dus_layers
    frozen_dus_params_m = frozen_dus_layers * 12.6
    dus_dtype_bytes = 4.0 if params["dus_dtype"] == "float32" else 2.0

    # Trainable: CA layers (unsharded) + FSDP-sharded unfrozen DUS layers + EMATracker shadow
    sharded_dus_m = unfrozen_dus_params_m / chips
    eff_trainable_m = ca_params_m + sharded_dus_m
    # Optimizer states (16B/param on sharded/unsharded) + EMATracker float32 shadow (4B/param)
    mem_trainable_gb = ((eff_trainable_m * 16.0 + trainable_params_m * 4.0) * 1e6) / (1024 ** 3)
    # Frozen DUS backbone is replicated across chips unless sharded; with SPMD + BF16/FP32 buffers:
    mem_frozen_dus_gb = (frozen_dus_params_m * 1e6 * dus_dtype_bytes * 0.65) / (1024 ** 3)
    mem_frozen_enc_gb = ((100.0 + 233.0) * 1e6 * 2.0) / (1024 ** 3)

    # Dictionary buffer (latent_dict)
    dict_size = params["dict_size"]
    dict_dtype_bytes = 2.0 if params["dict_dtype"] == "bfloat16" else 4.0
    mem_dict_buffer_gb = (dict_size * hidden_dim * dict_dtype_bytes) / (1024 ** 3)

    mem_static_total_gb = mem_trainable_gb + mem_frozen_dus_gb + mem_frozen_enc_gb + mem_dict_buffer_gb

    # 2. Peak Projection / Vocabulary Chunking Memory (HLO Temp)
    # With SPMD batch sharding ('fsdp'), each chip computes sims_chunk on batch_per_chip
    dict_chunks = max(1, params["dict_chunks"])
    chunk_vocab = dict_size / dict_chunks
    mem_single_sims_gb = (batch_per_chip * seq_len * chunk_vocab * dict_dtype_bytes) / (1024 ** 3)
    mem_peak_projection_gb = mem_single_sims_gb * 2.0

    # 3. Activations & Backward HLO Program Reservation (Forward & Backward)
    # Crucial XLA detail: even if only k layers after CA are unfrozen, autograd MUST backpropagate
    # from layer 39 all the way down to the earliest trainable layer (CA at index 11 -> 29 layers: 11..39)!
    num_layers = params["num_layers"]
    active_bwd_layers = (num_layers - params["min_ca_layer_idx"]) if params["has_ca_layers"] else max(1, unfrozen_dus_layers)
    act_bytes = dus_dtype_bytes  # 4.0 bytes if self.dus is float32

    if params["use_gradient_checkpointing"]:
        mem_activations_gb = (batch_per_chip * seq_len * hidden_dim * 6.0 * act_bytes) / (1024 ** 3)
    else:
        # Per active backward layer in ModernBERT (D=1024, H=16, GeGLU 5376, 25% global / 75% sliding window):
        # Saved activations (norm, qkv, GeGLU gate/up, residual) + hybrid SDPA backward workspace:
        linear_act_per_layer = 4.65 * batch_per_chip * seq_len * hidden_dim * act_bytes
        attn_ws_per_layer = 0.68 * batch_per_chip * 16 * seq_len * seq_len * act_bytes
        ca_act_total = 3.0 * (3.5 * batch_per_chip * seq_len * hidden_dim * act_bytes + 0.8 * batch_per_chip * 16 * seq_len * seq_len * act_bytes) if params["has_ca_layers"] else 0.0
        mem_activations_gb = ((active_bwd_layers * (linear_act_per_layer + attn_ws_per_layer)) + ca_act_total) / (1024 ** 3)

    # 4. Anti-pattern Memory Penalties (Graph Breaks, Live Outputs, Device Backups, Diagnosis Cache)
    mem_penalties_gb = 0.0
    penalty_reasons = []
    if params["has_mid_step_loss_sync"]:
        p_sync = (batch_per_chip * seq_len * hidden_dim * active_bwd_layers * act_bytes * 0.65) / (1024 ** 3)
        mem_penalties_gb += p_sync
        penalty_reasons.append(f"Mid-step 'isfinite(loss)' sync splits Fwd/Bwd graph (+{p_sync:.2f} GB)")
    if params["has_outputs_after_mark_step"]:
        p_out = (3.0 * batch_per_chip * seq_len * hidden_dim * 4.0) / (1024 ** 3)
        mem_penalties_gb += p_out
        penalty_reasons.append(f"'outputs' dict kept alive across xm.mark_step() (+{p_out:.2f} GB)")
    if params["has_device_ema_backup"]:
        p_ema = (trainable_params_m * 1e6 * 4.0) / (1024 ** 3)
        mem_penalties_gb += p_ema
        penalty_reasons.append(f"EMATracker.apply_shadow clones backup on TPU instead of CPU (+{p_ema:.2f} GB)")
    if params["has_resume_after_fsdp"]:
        p_fsdp = 0.35
        mem_penalties_gb += p_fsdp
        penalty_reasons.append(f"Resume load_state_dict after FSDP / duplicate EMATracker (+{p_fsdp:.2f} GB)")
    if params["run_diagnosis_first"]:
        p_diag = 0.30
        mem_penalties_gb += p_diag
        penalty_reasons.append(f"run_diagnosis_first=True caches 3 diagnostic HLO programs (+{p_diag:.2f} GB)")

    # 5. XLA Runtime & Compiler Workspace Overhead
    mem_xla_runtime_gb = 1.00

    # Program Reservation (Dynamic HLO Workspace) vs Static Resident Memory
    program_reservation_gb = mem_activations_gb + mem_peak_projection_gb + mem_penalties_gb
    free_hbm_before_program_gb = target["hbm_usable"] - mem_static_total_gb - mem_xla_runtime_gb

    # Total Peak HBM Estimate
    total_peak_gb = mem_static_total_gb + program_reservation_gb + mem_xla_runtime_gb
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
        "dus_dtype": params["dus_dtype"],
        "active_bwd_layers": active_bwd_layers,
        "dict_chunks": dict_chunks,
        "dict_dtype": params["dict_dtype"],
        "mem_static_gb": mem_static_total_gb,
        "mem_trainable_gb": mem_trainable_gb,
        "mem_dict_buffer_gb": mem_dict_buffer_gb,
        "mem_peak_projection_gb": mem_peak_projection_gb,
        "mem_single_sims_gb": mem_single_sims_gb,
        "mem_activations_gb": mem_activations_gb,
        "mem_penalties_gb": mem_penalties_gb,
        "penalty_reasons": penalty_reasons,
        "mem_xla_runtime_gb": mem_xla_runtime_gb,
        "program_reservation_gb": program_reservation_gb,
        "free_hbm_before_program_gb": free_hbm_before_program_gb,
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
    print(f"Sequence Length (T):   {res['seq_len']} | Backbone Precision: {res['dus_dtype']} ({res['active_bwd_layers']} bwd layers)")
    print(f"Dictionary Chunking:   {res['dict_chunks']} chunk(s) (dtype: {res['dict_dtype']})")
    print(f"{'-'*65}")
    print(f"Estimated Breakdown:")
    print(f"  • Static Model & AdamW Buffers:    {res['mem_static_gb']:6.2f} GB")
    print(f"      (Trainable + AdamW: {res['mem_trainable_gb']:.2f} GB | Dict Buffer: {res['mem_dict_buffer_gb']:.2f} GB)")
    print(f"  • XLA Runtime Reserve & Workspace:  {res['mem_xla_runtime_gb']:6.2f} GB")
    print(f"  -> Free HBM for XLA Program:       {res['free_hbm_before_program_gb']:6.2f} GB")
    print(f"  • Activations & Bwd Workspace:     {res['mem_activations_gb']:6.2f} GB")
    print(f"  • Peak Projection Buffer (Sims):    {res['mem_peak_projection_gb']:6.2f} GB  (2x XLA fusion/all-reduce)")
    if res["mem_penalties_gb"] > 0:
        print(f"  • Graph/Lifecycle Leaks & Penalties:{res['mem_penalties_gb']:5.2f} GB")
        for r in res["penalty_reasons"]:
            print(f"      [!] {r}")
    print(f"  -> XLA Program Reservation Needed: {res['program_reservation_gb']:6.2f} GB")
    print(f"{'-'*65}")
    print(f"ESTIMATED PEAK HBM DEMAND:           {res['total_peak_gb']:6.2f} GB / {res['usable_hbm']:.2f} GB")

    diff = res['total_peak_gb'] - res['usable_hbm']
    if res["status"] == "DANGER":
        print(f"\n[DANGER / OOM RISK] Peak memory exceeds usable HBM by {diff:+.2f} GB!")
        print(f"  (XLA Program attempts to reserve {res['program_reservation_gb']:.2f} GB, but only {res['free_hbm_before_program_gb']:.2f} GB is free!)")
        print(f"Actionable Fixes:")
        if res['mem_peak_projection_gb'] > 3.0:
            print(f"  -> Critical: Increase dictionary chunks (e.g. .chunk(16) or .chunk(32)).")
            print(f"  -> Critical: Switch latent_dict and matmul to bfloat16.")
        if res['batch_size'] > 64:
            print(f"  -> Reduce global batch_size from {res['batch_size']} to 64.")
        for r in res["penalty_reasons"]:
            print(f"  -> Fix lifecycle leak: {r}")
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
