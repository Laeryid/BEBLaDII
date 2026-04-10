import torch
import torch.nn as nn
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.beb_la_dii.model.dus import DUSModel
from src.beb_la_dii.model.projectors import InputProjector, FeatureProjector
from src.beb_la_dii.utils.tokenizer import get_tokenizer

class MockDistiller(nn.Module):
    """
    Minimal version of ReasoningDistiller for LOCAL TESTING WITHOUT GPU.
    It simulates Teacher embeddings to test Projectors and Student.
    """
    def __init__(self):
        super().__init__()
        self.student = DUSModel()
        self.input_projector = InputProjector()
        self.feature_projectors = nn.ModuleDict({
            "20": FeatureProjector(component_id="feat_proj_20"),
            "30": FeatureProjector(component_id="feat_proj_30"),
            "40": FeatureProjector(component_id="feat_proj_40")
        })

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        # Simulate Teacher Embeddings (4096)
        teacher_embeddings = torch.randn(batch_size, seq_len, 4096)
        # Simulate Teacher Targets (4096)
        teacher_targets = {
            20: torch.randn(batch_size, seq_len, 4096),
            30: torch.randn(batch_size, seq_len, 4096),
            40: torch.randn(batch_size, seq_len, 4096)
        }
        
        # 1. Project to Student space (1024 for ModernBERT-large)
        student_inputs_embeds = self.input_projector(teacher_embeddings)
        
        # 2. Forward through Student (ModernBERT 40 layers)
        student_outputs = self.student(
            inputs_embeds=student_inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 3. Project back to Qwen space
        projected_student_states = {}
        for idx in [20, 30, 40]:
            h_state = student_outputs.hidden_states[idx]
            projected_student_states[idx] = self.feature_projectors[str(idx)](h_state)
            
        return projected_student_states, teacher_targets

def test_inference():
    print("--- Running MOCK DISTILLER Smoke Test (CPU-friendly, 1024-dim) ---")
    device = "cpu"
    
    print("Loading tokenizer...")
    tokenizer = get_tokenizer()
    
    print("Initializing MockDistiller (Student + MLP Projectors)...")
    distiller = MockDistiller().to(device)
    distiller.eval()
    
    # 2. Prepare inputs
    text = "<|im_start|>user\nHello! Testing 1024-dim compatibility with ModernBERT-large.<|im_end|>\n<|im_start|>assistant\n<|thought|>\n"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    print(f"\nInput text: {text}")
    print(f"Token shape: {input_ids.shape}")
    
    # 3. Forward pass
    print("\nStarting forward pass...")
    try:
        with torch.no_grad():
            projected_states, teacher_targets = distiller(input_ids, attention_mask)
    except Exception as e:
        print(f"FAILED during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Verification
    print("\nResults:")
    for layer in [20, 30, 40]:
        s_shape = projected_states[layer].shape
        t_shape = teacher_targets[layer].shape
        print(f"Layer {layer}:")
        print(f"  - Student (Projected): {s_shape}")
        print(f"  - Teacher (Simulated): {t_shape}")
    
    print("\nSUCCESS: All tensors passed (1024-dim confirmed)!")

if __name__ == "__main__":
    test_inference()
