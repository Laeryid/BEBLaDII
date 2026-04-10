import os
import sys

# Добавляем путь к .know
sys.path.append(os.path.abspath(".know"))

from knowledge_mcp import validate_path

def test_security():
    print("Testing security jail...")
    
    try:
        # Test valid path
        p1 = validate_path("knowledge/KI_model_core.md")
        print(f"OK: Valid path allowed: {p1}")
    except Exception as e:
        print(f"FAIL: Valid path blocked: {e}")

    try:
        # Test traversal attack
        validate_path("../README.md")
        print("FAIL: Traversal attack allowed!")
    except PermissionError as e:
        print(f"OK: Traversal attack blocked: {e}")

    try:
        # Test absolute path attack
        target = os.path.abspath("README.md")
        validate_path(target)
        print("FAIL: Absolute path attack allowed!")
    except PermissionError as e:
        print(f"OK: Absolute path attack blocked: {e}")

if __name__ == "__main__":
    test_security()
