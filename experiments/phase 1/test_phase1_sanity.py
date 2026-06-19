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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_path", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to phase1_vae_step_X.pth")
    args = parser.parse_args()

    tester = VAESanityTester(args.qwen_path, args.ckpt_path)

    # 1. Тест Реконструкции
    tester.test_reconstruction("The quick brown fox jumps over the lazy dog.")
    tester.test_reconstruction("Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers.")

    # 2. Тест Интерполяции (SLERP)
    # Плавное перетекание одного смысла в другой
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
