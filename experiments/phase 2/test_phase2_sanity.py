import os
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

# Добавляем корень проекта в sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.beb_la_dii.model.vae import LatentEncoder
from src.beb_la_dii.model.modern_decoder import ModernLatentDecoder
from src.beb_la_dii.model.dus import DUSModel
from src.beb_la_dii.utils.loss import safe_normalize

def slerp(val, low, high):
    """Сферическая линейная интерполяция между двумя векторами (или тензорами векторов)."""
    low_norm = safe_normalize(low, dim=-1)
    high_norm = safe_normalize(high, dim=-1)
    
    # Косинусное сходство
    dot = (low_norm * high_norm).sum(dim=-1, keepdim=True).clamp(-0.99999, 0.99999)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    
    # Линейная интерполяция (fallback)
    lerp = (1.0 - val) * low + val * high
    
    # Поэлементная защита от деления на ноль
    safe_so = torch.where(so < 1e-5, torch.ones_like(so), so)
    slerp_res = torch.sin((1.0 - val) * omega) / safe_so * low + torch.sin(val * omega) / safe_so * high
    
    return torch.where(so < 1e-5, lerp, slerp_res)

class Phase2SanityTester:
    def __init__(self, qwen_path, phase1_ckpt_path, phase2_ckpt_path, dus_weights_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading Qwen Tokenizer from {qwen_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = 151643
            
        print(f"Loading Qwen Embeddings and LM Head...")
        qwen = AutoModelForCausalLM.from_pretrained(qwen_path, torch_dtype=torch.bfloat16)
        self.qwen_embed_weight = qwen.get_input_embeddings().weight.detach().clone().to(device)
        self.qwen_lm_head_weight = qwen.lm_head.weight.detach().clone().to(device)
        del qwen
        
        print(f"Loading Phase 1 VAE Encoder from {phase1_ckpt_path}...")
        self.encoder = LatentEncoder().to(device).to(torch.bfloat16)
        ckpt1 = torch.load(phase1_ckpt_path, map_location=device)
        self.encoder.load_state_dict(ckpt1["encoder"])
        self.encoder.eval()
        
        print(f"Loading Phase 2 Modern Decoder from {phase2_ckpt_path}...")
        
        if dus_weights_path and dus_weights_path.lower() == "none":
            dus_weights_path = None
            
        self.decoder = ModernLatentDecoder(
            latent_dim=1024,
            qwen_dim=1536,
            num_layers=3,
        ).to(device)
        
        ckpt2 = torch.load(phase2_ckpt_path, map_location=device)
        # Обратная совместимость на случай если ключи сохранены как decoder.xxx или без префикса
        state_dict = ckpt2["decoder"] if "decoder" in ckpt2 else ckpt2
        clean_state = {k.replace("decoder.", ""): v for k, v in state_dict.items()}
        self.decoder.load_state_dict(clean_state, strict=False)
        self.decoder.eval()
        
        print("Initialization complete!\n")

    @torch.no_grad()
    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        embeds = F.embedding(inputs.input_ids, self.qwen_embed_weight)
        z, _, _ = self.encoder(embeds)
        return inputs.input_ids, embeds, z

    @torch.no_grad()
    def decode_z(self, z):
        # Наш новый ModernLatentDecoder
        projected_embeds = self.decoder(z)
        
        # Проекция обратно в токены через сохраненную матрицу Qwen LM Head
        logits = F.linear(projected_embeds, self.qwen_lm_head_weight)
        
        pred_ids = logits.argmax(dim=-1)
        decoded_text = self.tokenizer.decode(pred_ids[0], skip_special_tokens=True)
        return decoded_text

    def test_reconstruction(self, text):
        print(f"--- RECONSTRUCTION TEST ---")
        print(f"ORIGINAL : {text}")
        _, _, z = self.encode_text(text)
        reconstructed = self.decode_z(z)
        print(f"DECODED  : {reconstructed}\n")

    def test_interpolation(self, text1, text2, steps=5):
        print(f"--- SPHERICAL INTERPOLATION (SLERP) ---")
        print(f"A: {text1}")
        print(f"B: {text2}")
        
        id1, _, z1 = self.encode_text(text1)
        id2, _, z2 = self.encode_text(text2)
        
        max_len = max(z1.size(1), z2.size(1))
        
        def pad_z(z_tensor, tgt_len):
            if z_tensor.size(1) < tgt_len:
                pad_size = tgt_len - z_tensor.size(1)
                pad_vec = z_tensor[:, -1:, :].repeat(1, pad_size, 1)
                return torch.cat([z_tensor, pad_vec], dim=1)
            return z_tensor
            
        z1_padded = pad_z(z1, max_len)
        z2_padded = pad_z(z2, max_len)
        
        for i in range(steps + 1):
            val = i / steps
            z_interp = slerp(val, z1_padded, z2_padded)
            decoded = self.decode_z(z_interp)
            print(f"Step {i}/{steps} (t={val:.2f}): {decoded}")
        print()

    def test_semantic_neighbors(self, sentences):
        print(f"--- SEMANTIC NEIGHBORS (MEAN POOLING) ---")
        z_means = []
        for s in sentences:
            _, _, z = self.encode_text(s)
            z_mean = safe_normalize(z.mean(dim=1), dim=-1)
            z_means.append(z_mean)
            
        z_stack = torch.cat(z_means, dim=0) # (N, D)
        sim_matrix = z_stack @ z_stack.T
        
        for i in range(len(sentences)):
            sims = sim_matrix[i].clone()
            sims[i] = -1.0
            best_idx = sims.argmax().item()
            score = sims[best_idx].item()
            print(f"Query: '{sentences[i]}'")
            print(f"Nearest: '{sentences[best_idx]}' (Cosine: {score:.3f})\n")

    @torch.no_grad()
    def test_prior_sampling(self, n_samples=16, seq_len=32):
        print("--- PRIOR SAMPLING TEST (hole detection) ---")
        latent_dim = 1024

        real_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Quantum computing harnesses the laws of quantum mechanics.",
            "It was a bright cold day in April, and the clocks were striking thirteen.",
            "Machine learning models require large amounts of training data.",
        ]
        real_ce_vals = []
        for text in real_texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding="max_length",
                                   truncation=True, max_length=seq_len).to(self.device)
            embeds = F.embedding(inputs.input_ids, self.qwen_embed_weight)
            z, _, _ = self.encoder(embeds)
            
            projected = self.decoder(z)
            logits = F.linear(projected, self.qwen_lm_head_weight)

            shift_logits = logits.contiguous()
            shift_labels = inputs.input_ids.contiguous()
            shift_mask = inputs.attention_mask.float()
            
            ce_raw = torch.nn.CrossEntropyLoss(reduction="none")(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            ).view(shift_labels.size())
            ce = (ce_raw * shift_mask).sum() / (shift_mask.sum() + 1e-6)
            real_ce_vals.append(ce.item())

        real_ce_mean = sum(real_ce_vals) / len(real_ce_vals)

        prior_ce_vals = []
        for _ in range(n_samples):
            z_prior = torch.randn(1, seq_len, latent_dim,
                                  dtype=torch.bfloat16, device=self.device)
            projected = self.decoder(z_prior)
            logits = F.linear(projected, self.qwen_lm_head_weight)

            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-9)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            prior_ce_vals.append(entropy.item())

        prior_entropy_mean = sum(prior_ce_vals) / len(prior_ce_vals)

        print(f"  Real text CE: {real_ce_mean:.4f}")
        print(f"  Prior z ~ N(0,I) entropy: {prior_entropy_mean:.4f}")

        vocab_size = 151936
        max_entropy = torch.log(torch.tensor(float(vocab_size))).item()
        normalized = prior_entropy_mean / max_entropy
        print(f"  Normalized prior entropy: {normalized:.3f}")

        gap = prior_entropy_mean - real_ce_mean
        print(f"  Gap (prior_entropy - real_CE): {gap:.4f}")
        if normalized > 0.85:
            verdict = "WARN: prior выглядит как равномерное распределение (collapsed decoder?)"
        elif normalized < 0.3:
            verdict = "WARN: слишком низкая энтропия — пространство prior коллапсировало"
        else:
            verdict = "OK: prior entropy в разумном диапазоне"
        print(f"  Verdict: {verdict}\n")

    @torch.no_grad()
    def test_neighbour_consistency(self, texts=None, noise_levels=(0.01, 0.05, 0.1, 0.3)):
        print("--- NEIGHBOUR CONSISTENCY TEST (smoothness) ---")
        if texts is None:
            texts = [
                "The sun rises in the east and sets in the west.",
                "Neural networks learn representations from data.",
                "The economy grew by three percent last quarter.",
            ]

        results_by_level = {eps: [] for eps in noise_levels}

        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            embeds = F.embedding(inputs.input_ids, self.qwen_embed_weight)
            z, _, _ = self.encoder(embeds)
            
            base_projected = self.decoder(z)
            base_flat = base_projected.view(-1, base_projected.size(-1))

            for eps in noise_levels:
                noise = eps * torch.randn_like(z)
                z_noisy = z + noise
                noisy_projected = self.decoder(z_noisy)
                noisy_flat = noisy_projected.view(-1, noisy_projected.size(-1))

                cos_sim = torch.nn.functional.cosine_similarity(
                    base_flat, noisy_flat, dim=-1
                ).mean().item()
                results_by_level[eps].append(cos_sim)

        print(f"  {'Noise (eps)':<14} {'Mean cos-sim':<16} {'Verdict'}")
        print(f"  {'-'*50}")
        all_ok = True
        for eps in noise_levels:
            mean_sim = sum(results_by_level[eps]) / len(results_by_level[eps])
            if eps <= 0.05:
                ok = mean_sim > 0.98
            elif eps <= 0.1:
                ok = mean_sim > 0.90
            else:
                ok = mean_sim > 0.70
            status = "OK" if ok else "WARN"
            if not ok:
                all_ok = False
            print(f"  eps={eps:<10.2f} {mean_sim:<16.4f} {status}")

        print(f"\n  Overall smoothness: {'GOOD' if all_ok else 'DEGRADED'}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_path", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--phase1_ckpt", type=str, default="checkpoints/phase1/phase1_vae_step_20000.pth")
    parser.add_argument("--phase2_ckpt", type=str, required=True, help="Path to phase2 decoder_step_X.pth")
    parser.add_argument("--dus_weights", type=str, default="storage/components/model/latentBERT/v1.0/weights.pt")
    args = parser.parse_args()

    # Делаем пути абсолютными, если они относительные
    phase1_ckpt = os.path.join(project_root, args.phase1_ckpt) if not os.path.isabs(args.phase1_ckpt) else args.phase1_ckpt
    phase2_ckpt = os.path.join(project_root, args.phase2_ckpt) if not os.path.isabs(args.phase2_ckpt) else args.phase2_ckpt
    dus_weights = os.path.join(project_root, args.dus_weights) if not os.path.isabs(args.dus_weights) else args.dus_weights

    tester = Phase2SanityTester(args.qwen_path, phase1_ckpt, phase2_ckpt, dus_weights)

    tester.test_reconstruction("The quick brown fox jumps over the lazy dog.")
    tester.test_reconstruction("Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers.")
    tester.test_reconstruction("Мама мыла раму, а папа чинил телевизор.")
    tester.test_reconstruction("Neural networks learn representations from data.")
    
    # ТЕСТ ДЛИННОГО АБЗАЦА (чтобы проверить, пропадает ли "мусор" после первых слов)
    tester.test_reconstruction(
        "Artificial intelligence and machine learning have revolutionized the way we interact with technology, "
        "enabling computers to understand natural language, recognize complex patterns in images, and make decisions "
        "with unprecedented accuracy. As these systems become more integrated into our daily lives, "
        "it is crucial to address the ethical implications and ensure that they are developed responsibly, "
        "with a focus on fairness, transparency, and human well-being."
    )
    
    tester.test_interpolation(
        "A little cat is sleeping comfortably on a soft pillow.",
        "A huge black dog barks aggressively at the passing car."
    )

    sentences = [
        "I love eating pizza with extra cheese.",
        "The theory of relativity was developed by Albert Einstein.",
        "Burgers and fries are my favorite fast food.",
        "Quantum physics describes the behavior of subatomic particles.",
        "It's raining heavily outside, I need an umbrella.",
        "The weather today is stormy and wet."
    ]
    tester.test_semantic_neighbors(sentences)

    tester.test_prior_sampling()
    tester.test_neighbour_consistency()
