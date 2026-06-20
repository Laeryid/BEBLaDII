import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

# Добавляем корень проекта в sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.beb_la_dii.model.vae import LatentEncoder, LatentDecoder
from src.beb_la_dii.utils.loss import safe_normalize

def slerp(val, low, high):
    """Сферическая линейная интерполяция между двумя векторами (или тензорами векторов)."""
    low_norm = safe_normalize(low, dim=-1)
    high_norm = safe_normalize(high, dim=-1)
    
    # Косинусное сходство
    dot = (low_norm * high_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    
    # Защита от деления на ноль, если векторы слишком близки
    if so.max().item() < 1e-5:
        return (1.0 - val) * low + val * high
        
    return torch.sin((1.0 - val) * omega) / so * low + torch.sin(val * omega) / so * high

class VAESanityTester:
    def __init__(self, qwen_path, ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading Qwen Tokenizer and Model from {qwen_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = 151643
            
        self.qwen = AutoModelForCausalLM.from_pretrained(qwen_path, torch_dtype=torch.bfloat16).to(device)
        self.qwen.eval()
        
        print(f"Loading VAE from {ckpt_path}...")
        self.encoder = LatentEncoder().to(device).to(torch.bfloat16)
        self.decoder = LatentDecoder().to(device).to(torch.bfloat16)
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.encoder.eval()
        self.decoder.eval()
        print("Initialization complete!\n")

    @torch.no_grad()
    def encode_text(self, text):
        """Возвращает эмбеддинги Qwen и латентный вектор z на сфере."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        embeds = self.qwen.get_input_embeddings()(inputs.input_ids)
        z, _, _ = self.encoder(embeds)
        z_normed = safe_normalize(z, dim=-1)
        return inputs.input_ids, embeds, z_normed

    @torch.no_grad()
    def decode_z(self, z):
        """Восстанавливает текст из латентного вектора z."""
        projected_embeds = self.decoder(z)
        outputs = self.qwen(inputs_embeds=projected_embeds)
        logits = outputs.logits
        
        # Берем argmax для Next-Token Prediction
        pred_ids = logits.argmax(dim=-1)
        # Так как logits[i] предсказывает token[i+1], результат сдвинут.
        # Декодируем как есть
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
        
        # Выравниваем размерности (паддинг до длины максимального)
        max_len = max(z1.size(1), z2.size(1))
        
        def pad_z(z_tensor, tgt_len):
            if z_tensor.size(1) < tgt_len:
                pad_size = tgt_len - z_tensor.size(1)
                # Берем вектор последнего токена и дублируем его для паддинга
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
            # Берем средний вектор предложения для кластеризации
            z_mean = safe_normalize(z.mean(dim=1), dim=-1)
            z_means.append(z_mean)
            
        z_stack = torch.cat(z_means, dim=0) # (N, D)
        sim_matrix = z_stack @ z_stack.T
        
        for i in range(len(sentences)):
            # Ищем ближайшего соседа (исключая себя)
            sims = sim_matrix[i].clone()
            sims[i] = -1.0
            best_idx = sims.argmax().item()
            score = sims[best_idx].item()
            print(f"Query: '{sentences[i]}'")
            print(f"Nearest: '{sentences[best_idx]}' (Cosine: {score:.3f})\n")

    @torch.no_grad()
    def test_prior_sampling(self, n_samples=16, seq_len=32):
        """
        Тест покрытия пространства ('дыры').
        Сэмплируем z ~ N(0, I) и считаем CE относительно случайного текста.
        Если CE близко к CE на реальных данных — prior покрыт хорошо, дыр мало.
        Чем выше gap (prior_ce - real_ce), тем больше дыр.
        """
        print("--- PRIOR SAMPLING TEST (hole detection) ---")
        latent_dim = 1024  # LatentEncoder output dim

        # 1. CE на реальном тексте (baseline)
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
            embeds = self.qwen.get_input_embeddings()(inputs.input_ids)
            z, mu, logvar = self.encoder(embeds)
            projected = self.decoder(z)
            logits = self.qwen(inputs_embeds=projected).logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs.input_ids[..., 1:].contiguous()
            shift_mask = inputs.attention_mask[..., 1:].float()
            ce_raw = torch.nn.CrossEntropyLoss(reduction="none")(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            ).view(shift_labels.size())
            ce = (ce_raw * shift_mask).sum() / (shift_mask.sum() + 1e-6)
            real_ce_vals.append(ce.item())

        real_ce_mean = sum(real_ce_vals) / len(real_ce_vals)

        # 2. CE на prior z ~ N(0, I)
        prior_ce_vals = []
        for _ in range(n_samples):
            z_prior = torch.randn(1, seq_len, latent_dim,
                                  dtype=torch.bfloat16, device=self.device)
            projected = self.decoder(z_prior)
            logits = self.qwen(inputs_embeds=projected).logits

            # Для prior нет «правильного» текста — меряем энтропию самого распределения
            # (логарифм мягкого максимума = насколько модель «уверена»)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-9)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            prior_ce_vals.append(entropy.item())

        prior_entropy_mean = sum(prior_ce_vals) / len(prior_ce_vals)

        print(f"  Real text CE (mean over {len(real_texts)} samples): {real_ce_mean:.4f}")
        print(f"  Prior z ~ N(0,I) entropy (mean over {n_samples} samples): {prior_entropy_mean:.4f}")

        # Vocab size для Qwen2.5-1.5B = 151936
        vocab_size = self.qwen.config.vocab_size
        max_entropy = torch.log(torch.tensor(float(vocab_size))).item()
        normalized = prior_entropy_mean / max_entropy
        print(f"  Normalized prior entropy: {normalized:.3f}  (1.0 = uniform = max holes, 0.0 = collapsed)")

        gap = prior_entropy_mean - real_ce_mean
        print(f"  Gap (prior_entropy - real_CE): {gap:.4f}")
        if normalized > 0.85:
            verdict = "WARN: prior выглядит как равномерное распределение — возможны дыры или collapsed decoder"
        elif normalized < 0.3:
            verdict = "WARN: слишком низкая энтропия — пространство prior коллапсировало"
        else:
            verdict = "OK: prior entropy в разумном диапазоне — серьёзных дыр не обнаружено"
        print(f"  Verdict: {verdict}\n")
        return {"real_ce": real_ce_mean, "prior_entropy": prior_entropy_mean,
                "normalized_entropy": normalized, "gap": gap}

    @torch.no_grad()
    def test_neighbour_consistency(self, texts=None, noise_levels=(0.01, 0.05, 0.1, 0.3)):
        """
        Тест гладкости / непрерывности пространства.
        Добавляем шум разной величины к mu (детерминированной части z)
        и смотрим, как падает косинусное сходство декодированных эмбеддингов.
        Быстрое падение → пространство не гладкое.
        """
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
            embeds = self.qwen.get_input_embeddings()(inputs.input_ids)
            _, mu, _ = self.encoder(embeds)  # mu: (1, T, D)

            # Базовые декодированные эмбеддинги (без шума)
            base_projected = self.decoder(mu)
            base_flat = base_projected.view(-1, base_projected.size(-1))  # (T, 1536)

            for eps in noise_levels:
                noise = eps * torch.randn_like(mu)
                z_noisy = mu + noise
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

        print(f"\n  Overall smoothness: {'GOOD — space is locally continuous' if all_ok else 'DEGRADED — abrupt boundaries detected'}\n")
        return {eps: sum(v) / len(v) for eps, v in results_by_level.items()}

    def run_full_report(self, ckpt_path):
        """Запускает все тесты и печатает сводный отчёт."""
        print("=" * 60)
        print(f" PHASE 1 VAE FULL DIAGNOSTIC REPORT")
        print(f" Checkpoint: {ckpt_path}")
        print("=" * 60 + "\n")

        # Стандартные тесты
        self.test_reconstruction("The quick brown fox jumps over the lazy dog.")
        self.test_reconstruction(
            "Quantum computing is a rapidly-emerging technology "
            "that harnesses the laws of quantum mechanics to solve "
            "problems too complex for classical computers."
        )
        self.test_interpolation(
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
        self.test_semantic_neighbors(sentences)

        # Новые тесты
        prior_stats = self.test_prior_sampling()
        smooth_stats = self.test_neighbour_consistency()

        print("=" * 60)
        print(" SUMMARY")
        print("=" * 60)
        print(f"  Prior normalized entropy : {prior_stats['normalized_entropy']:.3f}  (target: 0.30–0.85)")
        print(f"  Prior gap (entropy-CE)   : {prior_stats['gap']:.4f}")
        print(f"  Smoothness @ eps=0.01    : {smooth_stats[0.01]:.4f}  (target: >0.98)")
        print(f"  Smoothness @ eps=0.05    : {smooth_stats[0.05]:.4f}  (target: >0.90)")
        print(f"  Smoothness @ eps=0.10    : {smooth_stats[0.1]:.4f}  (target: >0.70)")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_path", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to phase1_vae_step_X.pth")
    args = parser.parse_args()

    tester = VAESanityTester(args.qwen_path, args.ckpt_path)

    if "--full" in sys.argv:
        # Полный отчёт: все тесты + новые диагностики
        tester.run_full_report(args.ckpt_path)
    else:
        # 1. Тест Реконструкции
        tester.test_reconstruction("The quick brown fox jumps over the lazy dog.")
        tester.test_reconstruction("Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers.")

        # 2. Тест Интерполяции (SLERP)
        tester.test_interpolation(
            "A little cat is sleeping comfortably on a soft pillow.",
            "A huge black dog barks aggressively at the passing car."
        )

        # 3. Тест Семантических Кластеров
        sentences = [
            "I love eating pizza with extra cheese.",
            "The theory of relativity was developed by Albert Einstein.",
            "Burgers and fries are my favorite fast food.",
            "Quantum physics describes the behavior of subatomic particles.",
            "It's raining heavily outside, I need an umbrella.",
            "The weather today is stormy and wet."
        ]
        tester.test_semantic_neighbors(sentences)

        # 4. Тест покрытия (дыры в пространстве)
        tester.test_prior_sampling()

        # 5. Тест гладкости (непрерывность)
        tester.test_neighbour_consistency()
