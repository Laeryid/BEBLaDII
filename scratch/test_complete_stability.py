import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("experiments/phase 1")
from train_phase1 import GrassmannianPhase1Model

def test_complete_stability():
    # Загружаем чекпоинт шага 800
    ckpt_path = "experiments/phase 1/checkpoints/phase1_best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    cleaned_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model = GrassmannianPhase1Model(vocab_size=131072)
    model.load_state_dict(cleaned_sd, strict=True)
    model.train()

    # Имитируем дальнейший рост весов dec_smooth_layers до шага 922 (умножаем веса на 1.3)
    with torch.no_grad():
        for layer in model.dec_smooth_layers:
            layer.gate_proj.weight.mul_(1.3)
            layer.up_proj.weight.mul_(1.3)
            layer.down_proj.weight.mul_(1.3)

    B, N, K, D = 2, 64, 8, 128
    device = torch.device("cpu")
    V_cell = torch.randn(B, N, K, D, device=device) * 50.0

    print("--- 1. ТЕСТ С ТЕКУЩИМ КОДОМ (FP16 autocast) ---")
    with torch.autocast(device_type="cpu", dtype=torch.float16):
        logits = model.decode(V_cell)
        print("Logits finite:", torch.isfinite(logits).all().item())

    print("\n--- 2. ТЕСТ В BFLOAT16 (как в PyTorch 2.6 на Kaggle T4 изначально) ---")
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits_bf16 = model.decode(V_cell)
        print("Logits bf16 finite:", torch.isfinite(logits_bf16).all().item(), "max:", logits_bf16.abs().max().item())

    print("\n--- 3. ТЕСТ С ROBUST RMSNORM И SWIGLU В FP16 ---")
    # Патчим RMSNorm и SwiGLU
    def robust_rmsnorm_forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = torch.nan_to_num(x.float(), nan=0.0, posinf=65000.0, neginf=-65000.0)
        var = torch.mean(x_f ** 2, dim=-1, keepdim=True)
        normed = x_f * torch.rsqrt(var + self.eps)
        return (normed * self.weight.float()).to(x.dtype)

    def robust_modernbert_forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                                  attention_mask=None) -> torch.Tensor:
        B, T, C = x.shape
        normed_x = self.attn_norm(x)
        q = self.q_proj(normed_x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed_x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed_x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        from train_phase1 import apply_rotary_pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.o_proj(attn_out)

        normed_mlp = self.mlp_norm(x)
        gate = F.silu(self.gate_proj(normed_mlp).float())
        up = self.up_proj(normed_mlp).float()
        gate_up = (gate * up).clamp(-65000.0, 65000.0).to(x.dtype)
        mlp_out = self.down_proj(gate_up)
        x = x + mlp_out
        return x

    from train_phase1 import RMSNorm, ModernBERTLayer
    RMSNorm.forward = robust_rmsnorm_forward
    ModernBERTLayer.forward = robust_modernbert_forward

    with torch.autocast(device_type="cpu", dtype=torch.float16):
        logits_robust = model.decode(V_cell)
        print("Logits robust fp16 finite:", torch.isfinite(logits_robust).all().item(), "max:", logits_robust.abs().max().item())

if __name__ == "__main__":
    test_complete_stability()
