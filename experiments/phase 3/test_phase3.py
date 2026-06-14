import os
import sys
import json
import torch
import torch.nn.functional as F
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from beb_la_dii.models.dus import DUSModel
from beb_la_dii.models.projectors import InputProjector, OutputProjector

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-phase2", type=str, required=True, help="Путь к весам DUS и IP (например, latest_checkpoint.pt)")
    parser.add_argument("--checkpoint-op", type=str, required=True, help="Путь к весам OutputProjector (например, OP_tau_005_layers_3.pt)")
    parser.add_argument("--dictionaries", type=str, required=True, help="Путь к папке с D_X0.pt и token_map.json")
    parser.add_argument("--text", type=str, default=" mathematics", help="Входной текст для теста")
    parser.add_argument("--cycles", type=int, default=5, help="Количество циклов рекурсии")
    parser.add_argument("--top-k", type=int, default=5, help="Сколько соседей показывать")
    parser.add_argument("--op-layers", type=int, default=3, help="Количество слоев в OP")
    return parser.parse_args()

def get_nearest_neighbors(query_vec, D_X0, token_map, top_k=5):
    # query_vec: [1, 1024]
    # D_X0: [V, 1024]
    query_norm = F.normalize(query_vec.float(), p=2, dim=-1)
    dict_norm = F.normalize(D_X0.float(), p=2, dim=-1)
    
    sims = torch.matmul(query_norm, dict_norm.T).squeeze(0) # [V]
    vals, idxs = torch.topk(sims, top_k)
    
    res = []
    for val, idx in zip(vals, idxs):
        token_id = token_map[idx.item()]
        res.append((token_id, val.item()))
    return res

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используем устройство: {device}")

    # 1. Токенизатор и Qwen
    print("Загрузка Qwen...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", device_map="cpu", torch_dtype=torch.bfloat16)
    qwen_embed = qwen.model.embed_tokens.to(device)
    for p in qwen_embed.parameters(): p.requires_grad = False
    del qwen

    # 2. Модели BEBLaDII
    print("Инициализация DUS, IP, OP...")
    student = DUSModel.from_scratch(config={"base_model_id": "answerdotai/ModernBERT-large", "target_layers": 40}).to(device)
    input_projector = InputProjector.from_scratch().to(device)
    output_projector = OutputProjector.from_scratch(config={"num_layers": args.op_layers}).to(device)
    
    # Загрузка весов Фазы 2 (DUS + IP)
    print(f"Загрузка весов Phase 2 из {args.checkpoint_phase2}...")
    sd2 = torch.load(args.checkpoint_phase2, map_location="cpu")
    if "model_state_dict" in sd2: sd2 = sd2["model_state_dict"]
    
    cleaned = {}
    for key, v in sd2.items():
        k_clean = key.replace("_orig_module.", "").replace("module.", "")
        cleaned[k_clean] = v

    student_sd, ip_sd = {}, {}
    for key, v in cleaned.items():
        if key.startswith("student.model."): student_sd[key[len("student.model."):]] = v
        elif key.startswith("model.layers.") or key.startswith("model.embeddings") or key.startswith("model.final_norm"): student_sd[key[len("model."):]] = v
        if key.startswith("input_projector."): ip_sd[key[len("input_projector."):]] = v
        elif key.startswith("proj.") or key.startswith("mu_head.") or key.startswith("logvar_head."): ip_sd[key] = v

    if student_sd: student.model.load_state_dict(student_sd, strict=False)
    if ip_sd: input_projector.load_state_dict(ip_sd, strict=False)

    # Загрузка весов Фазы 3 (OP)
    print(f"Загрузка весов Phase 3 из {args.checkpoint_op}...")
    sd3 = torch.load(args.checkpoint_op, map_location="cpu")
    if "model_state_dict" in sd3: sd3 = sd3["model_state_dict"]
    output_projector.load_state_dict(sd3, strict=False)

    student.eval()
    input_projector.eval()
    output_projector.eval()

    # 3. Словари
    print("Загрузка словарей...")
    D_X0 = torch.load(os.path.join(args.dictionaries, "D_X0.pt"), map_location=device)
    with open(os.path.join(args.dictionaries, "token_map.json"), "r") as f:
        token_map = json.load(f)

    print("\n" + "="*60)
    print(f"Тест: '{args.text}'")
    print("="*60)

    # Подготовка входа
    anchor_text = "<|thought|>"
    full_text = anchor_text + args.text
    input_ids = tokenizer.encode(full_text, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], device=device)
    
    print(f"Токены: {input_ids}")
    for tid in input_ids:
        print(f"  {tid} -> '{tokenizer.decode([tid])}'")

    with torch.no_grad():
        # Базовый X0
        qwen_emb = qwen_embed(input_tensor)
        X0, _, _ = input_projector(qwen_emb)
        
        current_X = X0
        
        for cycle in range(args.cycles + 1):
            if cycle == 0:
                print("\n--- Cycle 0: Прямой поиск X0 в D_X0 (Sanity словаря) ---")
                vec_to_search = X0[0, -1, :].unsqueeze(0)
            else:
                print(f"\n--- Cycle {cycle}: Прогон через DUS -> OP ---")
                # Внимание: DUS ожидает input_ids только для position_embeddings, 
                # но мы можем передать inputs_embeds напрямую.
                # ModernBERT требует attention_mask.
                mask = torch.ones((1, current_X.size(1)), device=device)
                
                # Прогоняем через DUS
                outputs = student(inputs_embeds=current_X, attention_mask=mask)
                L40 = outputs.last_hidden_state
                
                # Проецируем обратно в X0
                current_X = output_projector(L40)
                
                vec_to_search = current_X[0, -1, :].unsqueeze(0)
                
            # Ищем соседей для последнего токена
            neighbors = get_nearest_neighbors(vec_to_search, D_X0, token_map, top_k=args.top_k)
            for rank, (tid, sim) in enumerate(neighbors, 1):
                token_str = tokenizer.decode([tid])
                print(f"  [{rank}] '{token_str}' \t(id={tid}, cos={sim:.3f})")

if __name__ == "__main__":
    main()
