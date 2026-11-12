import sys
import torch
import torch.nn.functional as F

sys.path.append("experiments/phase 1")
from train_phase1 import GrassmannianPhase1Model

def test_fixed_gram_mlp():
    ckpt_path = "experiments/phase 1/checkpoints/phase1_best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    cleaned_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model = GrassmannianPhase1Model(vocab_size=131072)
    model.load_state_dict(cleaned_sd, strict=True)
    model.eval()

    B, N, K, D = 2, 32, 8, 128
    torch.manual_seed(42)
    V_cell = torch.randn(B, N, K, D)
    V_cell, _ = torch.linalg.qr(V_cell.view(B * N, D, K))
    V_cell = V_cell.transpose(-2, -1).view(B, N, K, D)

    # Заменяем forward unpacker на float32-вычисление gram_mlp
    def forward_safe_unpacker(self, V_cell):
        B, N, K, D = V_cell.shape
        target_dtype = V_cell.dtype

        Q = self.q_norm(self.query_slots).unsqueeze(0).unsqueeze(0).expand(B, N, self.num_slots, D).to(target_dtype)
        K_mat = self.k_norm(V_cell)
        V_mat = V_cell

        scores = torch.matmul(Q, K_mat.transpose(-2, -1)) / (D ** 0.5)
        attn = F.softmax(scores.float(), dim=-1).to(target_dtype)
        h_attn = torch.matmul(attn, V_mat)

        # G и gram_mlp строго в float32
        V_f32 = V_cell.float()
        G_f32 = torch.matmul(V_f32, V_f32.transpose(-2, -1))
        G_flat_f32 = G_f32.view(B, N, 64)

        mlp_w0 = self.gram_mlp[0].weight.float()
        mlp_b0 = self.gram_mlp[0].bias.float() if self.gram_mlp[0].bias is not None else None
        h1 = F.silu(F.linear(G_flat_f32, mlp_w0, mlp_b0))
        mlp_w2 = self.gram_mlp[2].weight.float()
        mlp_b2 = self.gram_mlp[2].bias.float() if self.gram_mlp[2].bias is not None else None
        h2 = F.linear(h1, mlp_w2, mlp_b2)
        norm_w = self.gram_mlp[3].weight.float()
        variance = h2.pow(2).mean(-1, keepdim=True)
        h_normed = h2 * torch.rsqrt(variance + 1e-6) * norm_w
        h_gram = h_normed.to(target_dtype).unsqueeze(2).expand(B, N, self.num_slots, 128)

        h_comb = torch.cat([h_attn, h_gram], dim=-1)
        h_out = self.out_norm(self.out_proj(h_comb))
        return h_out.view(B, N * self.num_slots, self.hidden_size)

    import types
    model.unpacker.forward = types.MethodType(forward_safe_unpacker, model.unpacker)

    print("--- 1. Testing FP32 decode ---")
    with torch.no_grad():
        logits_fp32 = model.decode(V_cell)
        print("FP32 logits finite:", torch.isfinite(logits_fp32).all().item(), "min/max:", logits_fp32.min().item(), logits_fp32.max().item())

    print("\n--- 2. Testing pure FP16 model decode ---")
    model.half()
    with torch.no_grad():
        logits_f16 = model.decode(V_cell.half())
        print("FP16 logits finite:", torch.isfinite(logits_f16).all().item(), "min/max:", logits_f16.float().min().item(), logits_f16.float().max().item())

if __name__ == "__main__":
    test_fixed_gram_mlp()
