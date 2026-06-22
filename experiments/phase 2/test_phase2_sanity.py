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
        self.decoder = ModernLatentDecoder(
            latent_dim=1024, 
            qwen_dim=1536, 
            num_layers=3, 
            dus_weights_path=dus_weights_path
        ).to(device).to(torch.bfloat16)
        
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
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
    
    tester.test_interpolation(
        "A little cat is sleeping comfortably on a soft pillow.",
        "A huge black dog barks aggressively at the passing car."
    )
