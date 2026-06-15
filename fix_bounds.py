import re

with open("experiments/phase 3/build_dictionaries.py", "r", encoding="utf-8") as f:
    code = f.read()

old_str = """            for rank, (idx, sim) in enumerate(zip(topk.indices.tolist(), topk.values.tolist())):
                marker = "✓" if idx == tok_id else " "
                print(f"    {marker} [{rank+1}] {repr(token_map[idx]):20s} cos={sim:.3f}  (id={idx})")"""

new_str = """            for rank, (idx, sim) in enumerate(zip(topk.indices.tolist(), topk.values.tolist())):
                marker = "✓" if idx == tok_id else " "
                tok_text = repr(token_map[idx]) if idx < len(token_map) else "<OUT_OF_VOCAB>"
                print(f"    {marker} [{rank+1}] {tok_text:20s} cos={sim:.3f}  (id={idx})")"""

code = code.replace(old_str, new_str)

with open("experiments/phase 3/build_dictionaries.py", "w", encoding="utf-8") as f:
    f.write(code)

with open("experiments/phase 3/test_phase3.py", "r", encoding="utf-8") as f:
    test_code = f.read()

old_test1 = """                for rank, (idx, sim) in enumerate(zip(topk_l40.indices.tolist(), topk_l40.values.tolist())):
                    marker = "✓" if idx == input_ids[-1] else " "
                    print(f"    {marker} [{rank+1}] {repr(token_map[idx]):20s} cos={sim:.3f}  (id={idx})")"""

new_test1 = """                for rank, (idx, sim) in enumerate(zip(topk_l40.indices.tolist(), topk_l40.values.tolist())):
                    marker = "✓" if idx == input_ids[-1] else " "
                    tok_text = repr(token_map[idx]) if idx < len(token_map) else "<OUT_OF_VOCAB>"
                    print(f"    {marker} [{rank+1}] {tok_text:20s} cos={sim:.3f}  (id={idx})")"""

old_test2 = """                for rank, (idx, w) in enumerate(zip(topk_soft.indices.tolist(), alpha.tolist())):
                    print(f"    [{rank+1}] {repr(token_map[idx]):20s} weight={w:.4f}  (id={idx})")"""

new_test2 = """                for rank, (idx, w) in enumerate(zip(topk_soft.indices.tolist(), alpha.tolist())):
                    tok_text = repr(token_map[idx]) if idx < len(token_map) else "<OUT_OF_VOCAB>"
                    print(f"    [{rank+1}] {tok_text:20s} weight={w:.4f}  (id={idx})")"""

old_test3 = """                    for rank, (idx, sim) in enumerate(zip(topk.indices.tolist(), topk.values.tolist())):
                        marker = "✓" if idx == input_ids[-1] else " "
                        print(f"  {marker} [{rank+1}] {repr(token_map[idx]):20s} (id={idx}, cos={sim:.3f})")"""

new_test3 = """                    for rank, (idx, sim) in enumerate(zip(topk.indices.tolist(), topk.values.tolist())):
                        marker = "✓" if idx == input_ids[-1] else " "
                        tok_text = repr(token_map[idx]) if idx < len(token_map) else "<OUT_OF_VOCAB>"
                        print(f"  {marker} [{rank+1}] {tok_text:20s} (id={idx}, cos={sim:.3f})")"""

old_test4 = """            for rank, (idx, sim) in enumerate(zip(topk.indices.tolist(), topk.values.tolist())):
                marker = "✓" if idx == input_ids[-1] else " "
                print(f"    {marker} [{rank+1}] {repr(token_map[idx]):20s} cos={sim:.3f}  (id={idx})")"""

new_test4 = """            for rank, (idx, sim) in enumerate(zip(topk.indices.tolist(), topk.values.tolist())):
                marker = "✓" if idx == input_ids[-1] else " "
                tok_text = repr(token_map[idx]) if idx < len(token_map) else "<OUT_OF_VOCAB>"
                print(f"    {marker} [{rank+1}] {tok_text:20s} cos={sim:.3f}  (id={idx})")"""

test_code = test_code.replace(old_test1, new_test1).replace(old_test2, new_test2).replace(old_test3, new_test3).replace(old_test4, new_test4)

with open("experiments/phase 3/test_phase3.py", "w", encoding="utf-8") as f:
    f.write(test_code)
