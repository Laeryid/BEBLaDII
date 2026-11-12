import sys
import torch
import torch.nn.functional as F

sys.path.append("experiments/phase 1")
from train_phase1 import GrassmannianPhase1Model

def inspect_smooth_layers():
    ckpt_path = "experiments/phase 1/checkpoints/phase1_best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    cleaned_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model = GrassmannianPhase1Model(vocab_size=131072)
    model.load_state_dict(cleaned_sd, strict=True)
    model.train()

    print("=== АНАЛИЗ ВЕСОВ dec_smooth_layers НА ШАГЕ 800 ===")
    for i, layer in enumerate(model.dec_smooth_layers):
        print(f"\n--- dec_smooth_layers[{i}] ---")
        for name, param in layer.named_parameters():
            print(f"  {name:15s}: shape={str(list(param.shape)):15s} norm={param.norm().item():.3f} max={param.abs().max().item():.4f}")

    # Проверим синтетический прогон через dec_smooth_layers
    B, N, chunk_size = 2, 64, 4
    full_len = N * chunk_size
    device = torch.device("cpu")

    # Возьмем случайный h_unpacked с типичной нормой
    h_unpacked = torch.randn(B, full_len, 1024) * 3.0  # типичная дисперсия после unpacker
    cos_full, sin_full = model.rope(h_unpacked, full_len)

    print("\n=== ПРОГОН ЧЕРЕЗ dec_smooth_layers В FP16 ===")
    with torch.autocast(device_type="cpu", dtype=torch.float16):
        x = h_unpacked
        for i, layer in enumerate(model.dec_smooth_layers):
            normed_x = layer.attn_norm(x)
            q = layer.q_proj(normed_x).view(B, full_len, layer.num_heads, layer.head_dim).transpose(1, 2)
            k = layer.k_proj(normed_x).view(B, full_len, layer.num_heads, layer.head_dim).transpose(1, 2)
            v = layer.v_proj(normed_x).view(B, full_len, layer.num_heads, layer.head_dim).transpose(1, 2)
            print(f"Layer {i} q max: {q.abs().max().item():.2f}, k max: {k.abs().max().item():.2f}")

            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
            attn_proj = layer.o_proj(attn_out.transpose(1, 2).contiguous().view(B, full_len, 1024))
            x_after_attn = x + attn_proj
            print(f"Layer {i} x_after_attn max: {x_after_attn.abs().max().item():.2f}")

            normed_mlp = layer.mlp_norm(x_after_attn)
            gate = F.silu(layer.gate_proj(normed_mlp))
            up = layer.up_proj(normed_mlp)
            gate_up = gate * up
            print(f"Layer {i} gate max: {gate.abs().max().item():.2f}, up max: {up.abs().max().item():.2f}, gate*up max: {gate_up.abs().max().item():.2f}")

            mlp_out = layer.down_proj(gate_up)
            x = x_after_attn + mlp_out
            print(f"Layer {i} x_after_mlp max: {x.abs().max().item():.2f}, finite: {torch.isfinite(x).all().item()}")

if __name__ == "__main__":
    inspect_smooth_layers()
