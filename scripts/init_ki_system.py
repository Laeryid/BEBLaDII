import os
import json
import sys
import argparse
from pathlib import Path

def detect_venv():
    """Пытается найти venv и вернуть путь к python.exe."""
    common_names = [".venv", "venv", "env"]
    for name in common_names:
        py_exe = Path(name) / "Scripts" / "python.exe"
        if py_exe.exists():
            return str(py_exe)
    return sys.executable

def find_manifest_dir():
    """Ищет папку, содержащую doc_config.json."""
    for root, dirs, files in os.walk("."):
        if "doc_config.json" in files:
            return root
    return None

def init_ki_system():
    print("[*] Initializing Knowledge Infrastructure...")
    
    parser = argparse.ArgumentParser(description="Инициализация системы знаний.")
    parser.add_argument("--root", help="Название папки для знаний (например, .know)")
    parser.add_argument("--agents", help="Путь к файлу инструкций агентов (например, AGENTS.md)")
    parser.add_argument("--workflows", help="Путь к директории воркфлоу")
    args = parser.parse_args()

    # 1. Определяем корень знаний
    knowledge_root = args.root or find_manifest_dir() or ".know"
    print(f"[+] Knowledge root: {knowledge_root}")

    # 2. Определяем AGENTS.md
    agent_file = args.agents or ("AGENTS.md" if os.path.exists("AGENTS.md") else None)
    if not agent_file:
        agent_file = "AGENTS.md"
        print(f"[!] AGENTS.md not found, will use default name.")
    else:
        print(f"[+] Found agent instructions: {agent_file}")

    # 3. Определяем воркфлоу
    workflows_dir = args.workflows
    if not workflows_dir:
        for d in [".agent/workflows", ".github/workflows", "workflows"]:
            if os.path.isdir(d):
                workflows_dir = d
                break
    workflows_dir = workflows_dir or ".agent/workflows"
    print(f"[+] Workflows directory: {workflows_dir}")

    # 4. Детектируем venv
    venv_py = detect_venv()
    print(f"[+] Python interpreter: {venv_py}")

    # 5. Создаем ki_config.json
    config = {
        "paths": {
            "knowledge_root": knowledge_root,
            "agent_instructions": agent_file,
            "workflows_dir": workflows_dir,
            "venv_python": venv_py
        },
        "auto_resolve": True
    }

    with open("ki_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    print("\n[+] Success! ki_config.json created.")
    
    # 6. Проверка структуры
    os.makedirs(knowledge_root, exist_ok=True)
    if not os.path.exists(os.path.join(knowledge_root, "doc_config.json")):
        print(f"[!] Warning: doc_config.json not found in {knowledge_root}. You may need to run initial indexing.")
    
    # 7. Инструкции для IDE
    mcp_script = os.path.join(knowledge_root, "knowledge_mcp.py")
    print("\n[!] To activate KnowledgeManager in your IDE (Cursor/Windsurf), add this tool command:")
    print(f"    {venv_py} {mcp_script} --config ki_config.json")
    print("-" * 40)

if __name__ == "__main__":
    init_ki_system()
