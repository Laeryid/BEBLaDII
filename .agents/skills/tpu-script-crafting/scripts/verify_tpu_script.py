#!/usr/bin/env python3
"""
Level 2 Validator: Deep Semantic AST-Based TPU Analyzer
Parses Python Abstract Syntax Tree (AST) to detect dangerous PyTorch/XLA
and TPU anti-patterns with precise context (loop scopes, conditionals, arg types).
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any


class Finding:
    def __init__(self, code: str, severity: str, message: str, line: int, fix: str):
        self.code = code
        self.severity = severity  # 'CRITICAL', 'WARNING'
        self.message = message
        self.line = line
        self.fix = fix


class TPUASTVisitor(ast.NodeVisitor):
    def __init__(self, source_lines: List[str]):
        self.source_lines = source_lines
        self.findings: List[Finding] = []
        
        # Scopes and context tracking
        self.in_loop: int = 0
        self.loop_stack: List[ast.AST] = []
        self.in_master_check: bool = False
        self.has_dynamo_disable: bool = False
        self.has_ema_apply: bool = False
        self.in_val_function: bool = False
        self.has_val_eval: bool = False

    def visit_Assign(self, node: ast.Assign):
        # Check: os.environ["XLA_USE_BF16"] = "1"
        for target in node.targets:
            if isinstance(target, ast.Subscript):
                if isinstance(target.value, ast.Attribute) and target.value.attr == "environ":
                    slice_node = target.slice
                    key_val = None
                    if isinstance(slice_node, ast.Constant):
                        key_val = slice_node.value
                    if key_val == "XLA_USE_BF16":
                        if isinstance(node.value, ast.Constant) and str(node.value.value) == "1":
                            self.findings.append(Finding(
                                "XLA_USE_BF16_ENV",
                                "CRITICAL",
                                "Active 'os.environ[\"XLA_USE_BF16\"] = \"1\"' forces compiler BF16 truncation (ADR 057, 082).",
                                node.lineno,
                                "Remove XLA_USE_BF16=1. Keep master weights and optimizer buffers in torch.float32."
                            ))
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        self.in_loop += 1
        self.loop_stack.append(node)
        self.generic_visit(node)
        self.loop_stack.pop()
        self.in_loop -= 1

    def visit_While(self, node: ast.While):
        self.in_loop += 1
        self.loop_stack.append(node)
        self.generic_visit(node)
        self.loop_stack.pop()
        self.in_loop -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef):
        prev_val = self.in_val_function
        prev_has_apply = self.has_ema_apply
        prev_has_eval = self.has_val_eval
        prev_in_loop = self.in_loop
        prev_loop_stack = self.loop_stack

        # Function definition resets loop context for its body
        self.in_loop = 0
        self.loop_stack = []

        func_name_lower = node.name.lower()
        if any(w in func_name_lower for w in ["val", "eval"]):
            self.in_val_function = True
            self.has_ema_apply = False
            self.has_val_eval = False

        self.generic_visit(node)

        if self.in_val_function:
            if self.has_val_eval and not self.has_ema_apply:
                self.findings.append(Finding(
                    "EVALUATING_LIVE_NOT_EMA",
                    "WARNING",
                    f"Validation function '{node.name}' calls .eval() but never invokes ema.apply() / ema.restore() (ADR 080).",
                    node.lineno,
                    "Wrap validation loop in: ema.apply(model); try: ... finally: ema.restore(model)."
                ))

        self.in_val_function = prev_val
        self.has_ema_apply = prev_has_apply
        self.has_val_eval = prev_has_eval
        self.in_loop = prev_in_loop
        self.loop_stack = prev_loop_stack

    def visit_If(self, node: ast.If):
        # Track if we are inside a master ordinal check (e.g. if rank == 0 or xm.is_master_ordinal())
        test_str = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
        is_master = any(k in test_str for k in ["is_master", "rank == 0", "local_rank == 0"])
        
        # Check: dynamic random condition inside train loop (e.g. if torch.rand(1).item() < 0.5)
        if self.in_loop > 0:
            if any(k in test_str for k in ["torch.rand", "random.random", "torch.randint"]) and ".item()" in test_str:
                self.findings.append(Finding(
                    "DYNAMIC_RAND_CONDITION",
                    "CRITICAL",
                    f"Dynamic random condition with .item() inside training loop: '{test_str[:60]}' (ADR 079).",
                    node.lineno,
                    "Use deterministic in-graph execution (e.g. fixed self-conditioning mask sc_mask) or CPU sampling."
                ))

        prev_master = self.in_master_check
        if is_master:
            self.in_master_check = True

        self.generic_visit(node)
        self.in_master_check = prev_master

    def visit_Call(self, node: ast.Call):
        func_name = ""
        attr_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if isinstance(node.func.value, ast.Name):
                func_name = f"{node.func.value.id}.{attr_name}"
            else:
                func_name = attr_name

        # Check: torch._dynamo.disable()
        if "disable" in attr_name and "dynamo" in func_name:
            self.has_dynamo_disable = True

        # Check: EMA apply
        if "apply" in attr_name and any(k in func_name.lower() for k in ["ema", "shadow"]):
            self.has_ema_apply = True

        if attr_name == "eval":
            self.has_val_eval = True

        # 1. Check: .item() inside loops
        if attr_name == "item" and self.in_loop > 0:
            line_content = self.source_lines[node.lineno - 1] if node.lineno <= len(self.source_lines) else ""
            if any(k in line_content.lower() for k in ["loss", "grad", "norm", "cos", "gate", "sim", "metric"]):
                self.findings.append(Finding(
                    "ITEM_CALL_IN_LOOP",
                    "CRITICAL",
                    f"Synchronous .item() call inside loop: '{line_content.strip()[:70]}' triggers graph breaks (ADR 079).",
                    node.lineno,
                    "Batch scalars into torch.stack([...]) and retrieve via single .cpu().tolist() in xm.add_step_closure."
                ))

        # 2. Check: DataLoader drop_last
        if attr_name == "DataLoader" or func_name.endswith("DataLoader"):
            has_drop_last = False
            for kw in node.keywords:
                if kw.arg == "drop_last":
                    if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        has_drop_last = True
            if not has_drop_last:
                self.findings.append(Finding(
                    "MISSING_DROP_LAST",
                    "WARNING",
                    "DataLoader instantiated without explicit 'drop_last=True'. Final partial batch forces XLA recompilation.",
                    node.lineno,
                    "Add drop_last=True to DataLoader kwargs."
                ))

        # 3. Check: torch.save without master check
        if func_name == "torch.save" and not self.in_master_check:
            self.findings.append(Finding(
                "TORCH_SAVE_ON_TPU",
                "CRITICAL",
                "torch.save() invoked outside master rank check. Multi-process TPU runs will collide and corrupt checkpoints.",
                node.lineno,
                "Wrap in 'if xm.is_master_ordinal():' or use xm.save(..., master_only=True)."
            ))

        # 4. Check: lerp_ with scalar
        if attr_name == "lerp_":
            # Check weight argument
            has_scalar_weight = False
            for kw in node.keywords:
                if kw.arg == "weight":
                    if isinstance(kw.value, (ast.Constant, ast.Name, ast.Attribute)):
                        val_str = ast.unparse(kw.value) if hasattr(ast, "unparse") else ""
                        if not any(k in val_str.lower() for k in ["tensor", "shadow"]):
                            has_scalar_weight = True
            if len(node.args) >= 2 and not has_scalar_weight:
                if isinstance(node.args[1], ast.Constant):
                    has_scalar_weight = True
            if has_scalar_weight:
                self.findings.append(Finding(
                    "LERP_SCALAR_BUG",
                    "CRITICAL",
                    "Calling .lerp_() with scalar weight boxes the scalar into CPU tensor on XLA, causing crashes or leaks.",
                    node.lineno,
                    "Use param.data.sub_(diff * alpha_tensor) or mul_().add_() instead of scalar lerp_."
                ))

        # 5. Check: gradient_checkpointing_enable on ModernBERT
        if attr_name == "gradient_checkpointing_enable":
            has_preserve_false = False
            for kw in node.keywords:
                if kw.arg == "gradient_checkpointing_kwargs":
                    if isinstance(kw.value, ast.Dict):
                        for k, v in zip(kw.value.keys, kw.value.values):
                            if isinstance(k, ast.Constant) and k.value == "preserve_rng_state":
                                if isinstance(v, ast.Constant) and v.value is False:
                                    has_preserve_false = True
            if not has_preserve_false:
                self.findings.append(Finding(
                    "GRADIENT_CHECKPOINTING_MODERNBERT",
                    "CRITICAL",
                    "gradient_checkpointing_enable() without {'preserve_rng_state': False}. Breaks XLA or leaks 90+ GB HBM (ADR 005).",
                    node.lineno,
                    "Disable GC for ModernBERT under XLA, or pass gradient_checkpointing_kwargs={'preserve_rng_state': False}."
                ))

        # 6. Check: ema.shadow.update
        if attr_name == "update" and "shadow" in func_name:
            self.findings.append(Finding(
                "EMA_RESUME_SHADOW_UPDATE",
                "CRITICAL",
                "Direct dictionary update on ema.shadow overwrites sharded TPU tensors with host CPU tensors (ADR 080).",
                node.lineno,
                "Use in-place .copy_(): ema.shadow[k].copy_(v.to(device=ema.shadow[k].device, dtype=torch.float32))."
            ))

        self.generic_visit(node)


def audit_file_ast(file_path: Path) -> List[Finding]:
    source_text = file_path.read_text(encoding="utf-8", errors="replace")
    source_lines = source_text.splitlines()

    try:
        tree = ast.parse(source_text, filename=str(file_path))
    except SyntaxError as e:
        return [Finding("SYNTAX_ERROR", "CRITICAL", f"SyntaxError: {e.msg}", e.lineno or 0, "Fix syntax error.")]

    visitor = TPUASTVisitor(source_lines)
    visitor.visit(tree)
    return visitor.findings


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_tpu_script.py <path_to_script.py>", file=sys.stderr)
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Error: file '{target}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"==================================================")
    print(f"=== [Level 2] AST Semantic Audit: {target.name}")
    print(f"==================================================")

    findings = audit_file_ast(target)

    if not findings:
        print("[OK] No critical TPU anti-patterns detected via AST analysis!\n")
        sys.exit(0)

    criticals = 0
    warnings = 0
    for f in findings:
        if f.severity == "CRITICAL":
            criticals += 1
            prefix = "[CRITICAL]"
        else:
            warnings += 1
            prefix = f"[{f.severity}]"
        print(f"\n{prefix} {f.code} (Line {f.line})")
        print(f"  Message: {f.message}")
        print(f"  Fix    : {f.fix}")

    print(f"\nAST Audit complete: {criticals} CRITICAL, {warnings} WARNINGS.\n")
    sys.exit(2 if criticals > 0 else 0)


if __name__ == "__main__":
    main()
