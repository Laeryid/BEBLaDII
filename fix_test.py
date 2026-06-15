import re

with open("experiments/phase 3/test_phase3.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. current_X -> current_h
code = code.replace("current_X = X0", "current_h = None")

# 2. Cycle 1 logic
old_logic = """                if cycle == 1:
                    # На первом прогоне через DUS используем X0 (z)
                    dus_out = student.model(inputs_embeds=X0, attention_mask=attention_mask)
                    X40 = dus_out.last_hidden_state  # [1, seq_len, 1024]
                else:
                    # На последующих прогонах используем вывод OP
                    dus_out = student.model(inputs_embeds=current_X, attention_mask=attention_mask)
                    X40 = dus_out.last_hidden_state"""

new_logic = """                if cycle == 1:
                    current_h = X0  # На первом прогоне используем X0 из словаря (это h)
                
                # По ADR-033: h (пространство OP) конвертируется в mu перед подачей в DUS
                mu_X = input_projector.mu_head(current_h)
                dus_out = student.model(inputs_embeds=mu_X, attention_mask=attention_mask)
                X40 = dus_out.last_hidden_state  # [1, seq_len, 1024]"""
code = code.replace(old_logic, new_logic)

# 3. Add Z_hat_target computation
old_sanity = """                # --- Sanity check L40 ---
                l40_vec = X40[0, -1, :].unsqueeze(0)
                l40_norm = F.normalize(l40_vec, dim=-1)
                scores_l40 = (l40_norm @ D_L40_norm.T)
                topk_l40 = torch.topk(scores_l40[0], k=5)
                
                print(f"\\n  [Sanity L40] Поиск вектора L40 в D_L40_norm:")
                for rank, (idx, sim) in enumerate(zip(topk_l40.indices.tolist(), topk_l40.values.tolist())):
                    marker = "✓" if idx == input_ids[-1] else " "
                    print(f"    {marker} [{rank+1}] {repr(token_map[idx]):20s} cos={sim:.3f}  (id={idx})")"""

new_sanity = """                # --- Sanity check L40 ---
                l40_vec = X40[0, -1, :].unsqueeze(0)
                l40_norm = F.normalize(l40_vec, dim=-1)
                scores_l40 = (l40_norm @ D_L40_norm.T)
                topk_l40 = torch.topk(scores_l40[0], k=5)
                
                print(f"\\n  [Sanity L40] Поиск вектора L40 в D_L40_norm:")
                for rank, (idx, sim) in enumerate(zip(topk_l40.indices.tolist(), topk_l40.values.tolist())):
                    marker = "✓" if idx == input_ids[-1] else " "
                    print(f"    {marker} [{rank+1}] {repr(token_map[idx]):20s} cos={sim:.3f}  (id={idx})")
                
                # --- Расчет Z_hat_target ---
                tau = 0.007
                scores_soft = scores_l40 / tau
                topk_soft = torch.topk(scores_soft[0], k=6)
                alpha = F.softmax(topk_soft.values, dim=-1)
                top_dx0 = D_X0[topk_soft.indices]
                Z_hat_raw = (alpha.unsqueeze(-1) * top_dx0).sum(dim=0)
                L_expected = (alpha * top_dx0.norm(dim=-1)).sum()
                Z_hat_target = F.normalize(Z_hat_raw, dim=-1) * L_expected
                
                print(f"\\n  [Z_hat_target] Ожидаемая цель для OP (компоненты D_X0):")
                for rank, (idx, w) in enumerate(zip(topk_soft.indices.tolist(), alpha.tolist())):
                    print(f"    [{rank+1}] {repr(token_map[idx]):20s} weight={w:.4f}  (id={idx})")"""
code = code.replace(old_sanity, new_sanity)

# 4. Fix output projection
old_proj = """                # Проецируем обратно в X0
                OP_out = output_projector(X40)  # [1, seq_len, 1024]
                current_X = OP_out
                vec_to_search = OP_out[0, -1, :].unsqueeze(0)"""

new_proj = """                # Проецируем L40 обратно в h
                OP_out = output_projector(X40)  # [1, seq_len, 1024]
                current_h = OP_out
                vec_to_search = OP_out[0, -1, :].unsqueeze(0)
                
                # Сравниваем с целью
                sim_target = F.cosine_similarity(vec_to_search, Z_hat_target.unsqueeze(0)).item()
                norm_op = vec_to_search.norm().item()
                norm_target = Z_hat_target.norm().item()
                print(f"\\n  [OP vs Target] cos={sim_target:.4f} | norm(OP)={norm_op:.2f} | norm(Target)={norm_target:.2f}")"""
code = code.replace(old_proj, new_proj)

with open("experiments/phase 3/test_phase3.py", "w", encoding="utf-8") as f:
    f.write(code)
