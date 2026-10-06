#!/usr/bin/env python3
"""
Level 1 Validator: Syntax & Bytecode Compilation Checker
Validates Python scripts for SyntaxError, IndentationError, TabError,
and encoding issues before sending to remote TPU VMs.
"""

import sys
import py_compile
from pathlib import Path


def check_syntax(file_path: Path) -> bool:
    print(f"=== [Level 1] Checking syntax: {file_path.name} ===")
    try:
        py_compile.compile(str(file_path), doraise=True)
        print(f"[OK] Syntax and bytecode compilation PASSED for {file_path.name}\n")
        return True
    except py_compile.PyCompileError as e:
        print(f"[SYNTAX ERROR] Failed to compile {file_path.name}:", file=sys.stderr)
        print(f"  {e.msg}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error compiling {file_path.name}: {e}", file=sys.stderr)
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_tpu_syntax.py <path_to_script.py>", file=sys.stderr)
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Error: file '{target}' does not exist.", file=sys.stderr)
        sys.exit(1)

    success = check_syntax(target)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
