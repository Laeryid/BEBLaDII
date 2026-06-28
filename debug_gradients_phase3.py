import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.beb_la_dii.model.dus import create_latentbert
from src.beb_la_dii.model.vae import LatentEncoder

class BEBLaDIIPhase3Debug(nn.Module):
    def __init__(self, embed_model_path="Qwen/Qwen2.5-1.5B"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(embed_model_path)
        causal_model = AutoModelForCausalLM.from_pretrained(
            embed_model_path, torch_dtype=torch.float32
        )
        self.qwen_embeddings = causal_model.get_input_embeddings()
        self.qwen_embeddings.requires_grad_(False)
        self.lm_head_weight = causal_model.lm_head.weight.data

        self.encoder = LatentEncoder().to(torch.float32)
        self.encoder.eval()
        self.encoder.requires_grad_(False)

        self.dus = create_latentbert(
            model_id="answerdotai/ModernBERT-large",
            target_layers=40,
        )
        self.confidence_proj = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 1024),
        )

        self.gradient_norms = {}
        self._register_hooks()

    def _register_hooks(self):
        def make_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self.gradient_norms[name] = grad_output[0].norm().item()
            return hook

        for i, layer in enumerate(self.dus.layers):
            layer.register_full_backward_hook(make_hook(f"dus_layer_{i}"))
        
        self.dus.final_norm.register_full_backward_hook(make_hook("dus_final_norm"))
        self.confidence_proj.register_full_backward_hook(make_hook("confidence_proj"))

    def forward(self, input_ids, attention_mask, low_noise_amp=0.05):
        B, T = input_ids.size()
        with torch.no_grad():
            word_embeds = self.qwen_embeddings(input_ids)
            enc_out = self.encoder(word_embeds)
            z_clean = enc_out[0].detach()

            noise = torch.randn_like(z_clean)
            c_true = torch.rand(B, T, device=z_clean.device)
            c_true = c_true.masked_fill(~attention_mask.bool(), 1.0)
            
            c_true_exp = c_true.unsqueeze(-1)
            z_noisy = c_true_exp * z_clean + (1.0 - c_true_exp) * noise
            z_noisy = F.normalize(z_noisy, dim=-1)

        c_embed_raw = self.confidence_proj(c_true.unsqueeze(-1).float())
        c_embed = F.normalize(c_embed_raw, dim=-1) * 0.1
        dus_input = z_clean + c_embed

        dus_outputs = self.dus(
            inputs_embeds=dus_input,
            output_hidden_states=True,
        )
        
        pre_norm = dus_outputs.hidden_states[-1]
        clean_pre_norm = pre_norm - c_embed
        dus_final_raw = self.dus.final_norm(clean_pre_norm)
        
        dus_delta = dus_final_raw - z_noisy
        dus_final = F.normalize(dus_final_raw, dim=-1)
        dus_layer33 = dus_outputs.hidden_states[33]

        return {
            "z_clean": z_clean,
            "z_noisy": z_noisy,
            "c_true": c_true,
            "dus_final": dus_final,
            "dus_layer33": dus_layer33,
            "c_embed": c_embed,
            "attention_mask": attention_mask
        }

def safe_normalize(x, dim=-1):
    return F.normalize(x, dim=dim)

def pearson_corr(x, y):
    x_c = x - x.mean(dim=-1, keepdim=True)
    y_c = y - y.mean(dim=-1, keepdim=True)
    num = (x_c * y_c).sum(dim=-1)
    den = torch.sqrt((x_c.pow(2).sum(dim=-1) + 1e-8) * (y_c.pow(2).sum(dim=-1) + 1e-8))
    return num / den

def main():
    print("[*] Creating Debug Model on CPU...")
    model = BEBLaDIIPhase3Debug()
    
    checkpoint_path = r"C:\Experiments\BEBLaDII\experiments\phase3\planB_phase3_checkpoints_phase3_step_2000.pth"
    print(f"[*] Loading checkpoint {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    cleaned_state = {}
    for k, v in state["dus"].items():
        if k.startswith("_fsdp_wrapped_module."):
            cleaned_state["dus." + k.replace("_fsdp_wrapped_module.", "")] = v
        else:
            cleaned_state["dus." + k] = v
            
    for k, v in state["confidence_proj"].items():
        if k.startswith("_fsdp_wrapped_module."):
            cleaned_state["confidence_proj." + k.replace("_fsdp_wrapped_module.", "")] = v
        else:
            cleaned_state["confidence_proj." + k] = v
            
    model.load_state_dict(cleaned_state, strict=False)
    print("[*] Checkpoint loaded.")
    
    # Mocking Teacher (so we don't blow up 16GB RAM)
    # The teacher layer 23 outputs 4096 dim
    
    B, T = 2, 32
    input_ids = torch.randint(0, 150000, (B, T))
    attention_mask = torch.ones((B, T))
    
    print("[*] Running Forward Pass...")
    out = model(input_ids, attention_mask)
    
    z_clean = out["z_clean"]
    dus_final = out["dus_final"]
    c_true = out["c_true"]
    dus_layer33 = out["dus_layer33"]
    attn_f = out["attention_mask"].float()
    
    print("[*] Computing Loss...")
    gamma = 20.0
    w_c = torch.pow(c_true, gamma)
    w_n = torch.pow(1.0 - c_true, gamma)
    active_tokens = attn_f.sum()
    
    # Denoise
    cos_denoise = (dus_final * z_clean).sum(dim=-1)
    denoise_loss = ((1.0 - cos_denoise) * w_n * attn_f).sum() / active_tokens
    
    # Identity
    cos_identity = (dus_final * out["z_noisy"]).sum(dim=-1)
    identity_loss = ((1.0 - cos_identity) * w_c * attn_f).sum() / active_tokens
    
    # Mocking Teacher with REAL structured vectors from Qwen 1.5B (to simulate DeepSeek)
    print("[*] Generating structured teacher vectors...")
    with torch.no_grad():
        # Use the already loaded Qwen causal_model for this
        causal_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B", torch_dtype=torch.float32, output_hidden_states=True
        )
        causal_model.eval()
        tea_outputs = causal_model(input_ids)
        # Qwen 1.5B has 28 layers. Let's take a deep layer to simulate teacher layer 23.
        teacher_layer23 = tea_outputs.hidden_states[-4].detach()
        # The dim is 1536. We'll pad it to 4096 just to match the tensor shapes if needed,
        # but Pearson RKD uses Cosine Similarity, which is dimension-agnostic!
        # So we can just use the 1536d vectors directly for Pearson RKD.
        del causal_model # Free RAM
    
    def extract_windows(hidden_states, window_size=15, hop=5):
        B, T, D = hidden_states.shape
        windows = []
        for i in range(0, T - window_size + 1, hop):
            windows.append(hidden_states[:, i : i + window_size, :])
        if not windows:
            return None
        return torch.stack(windows, dim=1) 
        
    dus_w = extract_windows(dus_layer33)
    tea_w = extract_windows(teacher_layer23)
    
    dus_w_norm = safe_normalize(dus_w)
    tea_w_norm = safe_normalize(tea_w)
    
    S_dus = torch.matmul(dus_w_norm, dus_w_norm.transpose(-1, -2))
    S_tea = torch.matmul(tea_w_norm, tea_w_norm.transpose(-1, -2))
    
    mask = torch.triu(torch.ones((15, 15), dtype=torch.bool), diagonal=1)
    dus_triu = S_dus[:, :, mask]
    tea_triu = S_tea[:, :, mask]
    
    logic_loss = (1.0 - pearson_corr(dus_triu, tea_triu)).mean()
    
    loss = denoise_loss * 1.0 + identity_loss * 5.0 + logic_loss * 1.0
    print(f"Losses: Denoise={denoise_loss.item():.4f}, Identity={identity_loss.item():.4f}, Logic={logic_loss.item():.4f}")
    
    print("[*] Running Backward Pass...")
    loss.backward()
    
    print("\n==== GRADIENT NORMS ====")
    print(f"Confidence Proj: {model.gradient_norms.get('confidence_proj', 0.0):.6f}")
    for i in range(0, 40, 5):
        print(f"DUS Layer {i}: {model.gradient_norms.get(f'dus_layer_{i}', 0.0):.6f}")
    print(f"DUS Layer 33 (Logic): {model.gradient_norms.get('dus_layer_33', 0.0):.6f}")
    print(f"DUS Final Norm: {model.gradient_norms.get('dus_final_norm', 0.0):.6f}")

if __name__ == "__main__":
    main()
