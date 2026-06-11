"""
eval_l0_geometry.py
-------------------
Замер геометрических свойств латентного пространства l0 (выход Input Projector / mu-VAE).

В inference mode: z = mu, поэтому мы измеряем геометрию mu-векторов.

Метрики:
  - mu_norm       : среднее ‖μᵢ‖ по активным токенам
  - isotropy      : (Σsᵢ)² / (N · Σsᵢ²) на SVD сингулярных значений (выше = лучше)
  - rank1_ratio   : s₀² / Σsᵢ² (доля дисперсии в первом компоненте, ниже = лучше)
  - norm_cv       : std(‖μᵢ‖) / mean(‖μᵢ‖) — сферичность (ниже = лучше, <0.05 = сфера)
  - cov_off_diag  : VICReg-style off-diagonal covariance penalty (ниже = лучше)
  - mu_drift      : mean(μ²) по измерениям (центрированность, ниже = лучше)
  - var_drift     : mean((var_d - 1)²) по измерениям (близость к N(0,1))
"""

import sys
import os
import torch
import torch.nn as nn

# --- Пути ---
REPO_ROOT = r"C:\Experiments\BEBLaDII"
sys.path.insert(0, REPO_ROOT)

CKPT_PATH  = r"C:\Experiments\BEBLaDII\storage\experiments\20260607 Phase 2 Reasoning and Topology\latest_checkpoint.pt"
EMBEDDER_PATH = r"C:\Experiments\BEBLaDII\storage\components\model\teacher_embedder\weights.pt"
TOKENIZER_ID  = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Набор текстов для анализа (разнообразные, ~100 токенов каждый)
TEXTS = [
    "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms.",
    "Latent diffusion models operate in a compressed representation space, enabling efficient generation.",
    "Mathematics provides the foundation for understanding neural network optimization and convergence.",
    "Deep learning systems require careful initialization to avoid vanishing or exploding gradients.",
    "The geometry of representation spaces determines how well models can generalize to new tasks.",
    "Information theory connects entropy, compression, and the capacity of communication channels.",
    "Optimization landscapes in high dimensions exhibit saddle points rather than local minima.",
    "Tokenization strategies affect how language models perceive and process input sequences.",
    "Regularization techniques such as dropout and weight decay prevent overfitting in deep models.",
    "The attention mechanism computes weighted sums of value vectors based on query-key similarity.",
    "Contrastive learning methods train representations by pulling similar samples together in latent space.",
    "Knowledge distillation transfers information from a large teacher model to a smaller student model.",
    "Variational autoencoders learn to encode data into distributions rather than point estimates.",
    "The spectral properties of a matrix reveal how information is distributed across its dimensions.",
    "Isotropic distributions have equal variance in all directions, unlike anisotropic cone-like structures.",
    "Gradient flow through deep networks is governed by the chain rule of calculus.",
    "Batch normalization stabilizes training by normalizing activations within each mini-batch.",
    "Principal component analysis decomposes data variance into orthogonal directions of maximal spread.",
    "Cross-entropy loss measures the divergence between predicted and true probability distributions.",
    "Residual connections allow gradients to flow directly through skip paths in deep architectures.",
]


def compute_svd_metrics(vecs: torch.Tensor) -> dict:
    """
    vecs: (N, D) float32 — активные токены.
    Возвращает словарь с геометрическими метриками.
    """
    N, D = vecs.shape

    # Центрирование
    mean = vecs.mean(dim=0)           # (D,)
    mu_drift = mean.pow(2).mean().item()

    centered = vecs - mean            # (N, D)

    # Дисперсия по измерениям
    var = vecs.var(dim=0, unbiased=False)   # (D,)
    var_drift = (var - 1.0).pow(2).mean().item()

    # SVD (экономичная форма)
    try:
        _, s, _ = torch.linalg.svd(centered, full_matrices=False)
    except Exception as e:
        print(f"SVD failed: {e}")
        return {}

    s2 = s.pow(2)
    s2_sum = s2.sum().clamp(min=1e-12)

    isotropy   = (s.sum() ** 2) / (len(s) * s2_sum)
    rank1_ratio = s2[0] / s2_sum

    # Нормы и CV
    norms = vecs.norm(dim=-1)           # (N,)
    mu_norm = norms.mean().item()
    norm_cv = (norms.std() / norms.mean().clamp(min=1e-6)).item()

    # Cov off-diagonal (VICReg-style, на центрированных)
    # cov = (Z^T Z) / N  → берём только внедиагональные элементы
    if N > 1:
        Z = centered / (N ** 0.5)              # (N, D)
        cov = Z.T @ Z                           # (D, D)
        mask_off = ~torch.eye(D, dtype=torch.bool, device=vecs.device)
        cov_off = cov[mask_off].pow(2).sum() / D
        cov_off = cov_off.item()
    else:
        cov_off = float("nan")

    return {
        "mu_norm":     mu_norm,
        "norm_cv":     norm_cv,
        "isotropy":    isotropy.item(),
        "rank1_ratio": rank1_ratio.item(),
        "mu_drift":    mu_drift,
        "var_drift":   var_drift,
        "cov_off_diag": cov_off,
        "N_tokens":    N,
        "D":           D,
    }


def main():
    from transformers import AutoTokenizer
    from src.beb_la_dii.model.projectors import InputProjector

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Токенизатор
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    # 2. Teacher Embedder (только эмбеддинговый слой)
    print("Loading teacher embedder...")
    teacher_embedder = nn.Embedding(152064, 3584, dtype=torch.bfloat16)
    teacher_embedder.load_state_dict(torch.load(EMBEDDER_PATH, map_location="cpu"))
    teacher_embedder.to(device).eval()

    # 3. Input Projector
    print("Loading Input Projector from checkpoint...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    proj_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("_orig_module.", "").replace("module.", "")
        if clean_k.startswith("input_projector."):
            proj_dict[clean_k.replace("input_projector.", "")] = v

    print(f"  Extracted {len(proj_dict)} keys for InputProjector.")
    if len(proj_dict) == 0:
        print("  ERROR: No input_projector keys found! Check checkpoint structure.")
        print(f"  Sample keys: {list(state_dict.keys())[:10]}")
        return

    input_projector = InputProjector()
    result = input_projector.load_state_dict(proj_dict, strict=False)
    print(f"  Missing keys: {result.missing_keys}")
    print(f"  Unexpected keys: {result.unexpected_keys}")
    input_projector.to(device).to(torch.float32).eval()

    # 4. Прогон текстов и сбор mu-векторов
    print(f"\nProcessing {len(TEXTS)} texts...")
    all_mu = []

    inputs = tokenizer(TEXTS, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids     = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        teacher_embeds = teacher_embedder(input_ids).to(torch.float32)  # (B, T, 3584)
        z, mu, logvar  = input_projector(teacher_embeds)                # (B, T, 1024)

    # Собираем только активные токены
    mask = attention_mask.bool()
    active_mu = mu[mask].float()      # (N_active, 1024)
    active_z  = z[mask].float()       # (N_active, 1024) — в inference mode == mu

    print(f"  Total active tokens: {active_mu.shape[0]}")

    # 5. Замер
    print("\n" + "=" * 65)
    print("GEOMETRY OF l0 (mu-vectors = z in inference mode)")
    print("=" * 65)

    metrics = compute_svd_metrics(active_mu)
    for name, val in metrics.items():
        print(f"  {name:<20}: {val:.5f}" if isinstance(val, float) else f"  {name:<20}: {val}")

    # 6. Сравнение с диагностической таблицей KI
    print("\n" + "-" * 65)
    print("СРАВНЕНИЕ С ДИАГНОСТИЧЕСКОЙ ТАБЛИЦЕЙ (KI_metrics_reference.md)")
    print("-" * 65)
    thresholds = {
        "isotropy":    [(">0.10", 0.10, "Хорошо"), ("0.03-0.07", 0.03, "Удовл."), ("<0.03", 0.0, "Критично")],
        "rank1_ratio": [("<0.40", 0.40, "Хорошо"), ("0.60-0.85", 0.85, "Удовл."), (">0.90", 1.0, "Критично")],
        "norm_cv":     [("<0.05", 0.05, "Сфера"), ("0.05-0.15", 0.15, "Удовл."), (">0.15", 1.0, "Критично")],
    }
    iso   = metrics.get("isotropy", 0)
    r1    = metrics.get("rank1_ratio", 1)
    ncv   = metrics.get("norm_cv", 1)

    print(f"  isotropy    {iso:.4f}  →  {'✅ Хорошо' if iso > 0.10 else ('⚠️  Удовл.' if iso > 0.03 else '🚨 Критично')}")
    print(f"  rank1_ratio {r1:.4f}  →  {'✅ Хорошо' if r1 < 0.40 else ('⚠️  Удовл.' if r1 < 0.85 else '🚨 Критично')}")
    print(f"  norm_cv     {ncv:.4f}  →  {'✅ Сфера' if ncv < 0.05 else ('⚠️  Удовл.' if ncv < 0.15 else '🚨 Критично')}")
    print("=" * 65)

    # 7. Logvar статистика (диагностика VAE режима)
    active_logvar = logvar[mask].float()
    print(f"\nVAE diagnostics:")
    print(f"  logvar_mean  : {active_logvar.mean().item():.4f}")
    print(f"  logvar_std   : {active_logvar.std().item():.4f}")
    print(f"  sigma_mean   : {active_logvar.mul(0.5).exp().mean().item():.4f}  (σ = exp(0.5·logvar))")
    print(f"  signal/noise : {metrics.get('mu_norm', 0) / max(active_logvar.mul(0.5).exp().mean().item(), 1e-6):.2f}x")


if __name__ == "__main__":
    main()
