#!/usr/bin/env python3
"""
Level 3 Validator: CPU Smoke Test & Dry-Run Engine
Executes a TPU training script in a mocked XLA environment on CPU for 1-2 steps.
Verifies imports, forward pass, backward pass, loss calculation,
optimizer step, EMA pullback, and checkpoint saving without needing actual TPU hardware.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Paths
SKILL_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = SKILL_ROOT / "scripts"
FAKE_XLA_DIR = SCRIPTS_DIR / "fake_torch_xla"
WORKSPACE_ROOT = Path(os.getcwd()).resolve()


def run_pipeline(target_script: Path, timeout_sec: int = 60, extra_args: list = None) -> int:
    print(f"\n{'='*60}")
    print(f"=== [Level 1] Syntax Check")
    print(f"{'='*60}")
    syntax_script = SCRIPTS_DIR / "verify_tpu_syntax.py"
    res_syntax = subprocess.run([sys.executable, str(syntax_script), str(target_script)])
    if res_syntax.returncode != 0:
        print("[FAIL] Level 1 Syntax Check failed. Aborting smoke test.", file=sys.stderr)
        return 1

    print(f"\n{'='*60}")
    print(f"=== [Level 2] AST Semantic Audit")
    print(f"{'='*60}")
    ast_script = SCRIPTS_DIR / "verify_tpu_script.py"
    res_ast = subprocess.run([sys.executable, str(ast_script), str(target_script)])
    if res_ast.returncode != 0:
        print("[WARNING] Level 2 found critical issues or warnings above.")

    print(f"\n{'='*60}")
    print(f"=== [Level 3] CPU Dry-Run / Smoke Test")
    print(f"{'='*60}")

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    new_pythonpath = f"{FAKE_XLA_DIR}{os.pathsep}{WORKSPACE_ROOT}{os.pathsep}{existing_pythonpath}"
    env["PYTHONPATH"] = new_pythonpath
    env["WANDB_MODE"] = "offline"
    env["SMOKE_TEST"] = "1"
    env["MAX_STEPS"] = "2"
    env["VAL_STEPS"] = "1"
    env["PJRT_DEVICE"] = "CPU"
    env["PROJECT_ROOT"] = str(WORKSPACE_ROOT)
    
    # We pass args to limit runtime if script accepts standard cli flags
    cmd = [sys.executable, str(target_script)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"Executing: {' '.join(cmd)}")
    print(f"Mocked XLA injected from: {FAKE_XLA_DIR}")
    print("Running for initial steps...\n" + "-"*50)

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec
        )
        print(proc.stdout)
        if proc.returncode == 0:
            print("-"*50)
            print(f"[SUCCESS] Level 3 Smoke Test passed! Script executed cleanly on CPU mock.")
            
            print(f"\n{'='*60}")
            print(f"=== [Level 4] Memory Budget & OOM Risk Analysis")
            print(f"{'='*60}")
            mem_script = SCRIPTS_DIR / "estimate_tpu_memory.py"
            res_mem = subprocess.run([sys.executable, str(mem_script), str(target_script)])
            if res_mem.returncode != 0:
                print("[WARNING] Level 4 detected critical OOM risk or memory limit exceed.", file=sys.stderr)
                return res_mem.returncode
            return 0
        else:
            print(proc.stderr, file=sys.stderr)
            print("-"*50)
            print(f"[FAIL] Script exited with code {proc.returncode}.", file=sys.stderr)
            return proc.returncode
    except subprocess.TimeoutExpired as e:
        # If it reached timeout, output stdout to see how far it went
        if e.stdout:
            print(e.stdout)
        print(f"\n[TIMEOUT] Script timed out after {timeout_sec}s. Check if data loader or download stalled.", file=sys.stderr)
        return 124


def main():
    parser = argparse.ArgumentParser(description="Multi-Level TPU Script Verification (Syntax -> AST -> Dry-Run).")
    parser.add_argument("script", help="Path to TPU Python script to verify")
    parser.add_argument("--timeout", type=int, default=60, help="Smoke test timeout in seconds (default: 60)")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra arguments to pass to the script")

    args = parser.parse_args()
    target = Path(args.script).resolve()
    if not target.exists():
        print(f"Error: file '{target}' does not exist.", file=sys.stderr)
        sys.exit(1)

    code = run_pipeline(target, timeout_sec=args.timeout, extra_args=args.extra_args)
    sys.exit(code)


if __name__ == "__main__":
    main()
