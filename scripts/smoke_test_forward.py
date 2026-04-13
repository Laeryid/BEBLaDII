"""
Smoke-test: one forward pass through the full ReasoningDistiller pipeline.

Usage:
    .venv/Scripts/python.exe scripts/smoke_test_forward.py

What is tested:
  - InputProjector: (B, T, 3584) -> (B, T, 1024)
  - DUSModel (latentBERT 40 layers): inputs_embeds -> hidden_states
  - FeatureProjector x3: (B, T, 1024) -> (B, T, 3584)

What is NOT needed:
  - Real DeepSeek weights (replaced by mock)
  - Training / backward / loss
  - Dataset

Runs on CPU. No GPU required.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from unittest.mock import MagicMock
from types import SimpleNamespace

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
TEACHER_HIDDEN = 3584       # DeepSeek-R1-Distill-Qwen-7B (Qwen2.5 base)
STUDENT_HIDDEN = 1024       # ModernBERT-large
STUDENT_LAYERS = 40         # DUS target layers
BATCH_SIZE = 1
SEQ_LEN = 512
DEVICE = "cpu"

print("=" * 60)
print("BEBLaDII -- Smoke Test: Forward Pass")
print("=" * 60)
print(f"  Teacher hidden size : {TEACHER_HIDDEN}")
print(f"  Student hidden size : {STUDENT_HIDDEN}")
print(f"  Student layers      : {STUDENT_LAYERS}")
print(f"  Batch / Seq len     : {BATCH_SIZE} / {SEQ_LEN}")
print(f"  Device              : {DEVICE}")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Import project components
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/4] Importing project components...")

from src.beb_la_dii.model.projectors import InputProjector, FeatureProjector
from src.beb_la_dii.model.dus import create_latentbert

print("  OK: projectors, dus")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Create mock teacher
# Simulates DeepSeek forward(): returns hidden_states of the correct shape.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Building mock teacher (no real weights needed)...")

def make_mock_teacher(hidden: int, num_layers: int, batch: int, seq: int, device: str):
    """
    Returns an object whose __call__ returns a structure identical to
    AutoModelForCausalLM output with output_hidden_states=True.
    """
    # num_layers + 1: [embedding] + [layer_1, ..., layer_N]
    fake_hidden_states = tuple(
        torch.randn(batch, seq, hidden, device=device)
        for _ in range(num_layers + 1)
    )
    output = SimpleNamespace(hidden_states=fake_hidden_states)
    mock = MagicMock()
    mock.return_value = output
    mock.parameters = MagicMock(return_value=iter([]))
    return mock

# DeepSeek-R1-Distill-Qwen-7B has 28 transformer blocks
TEACHER_TRANSFORMER_LAYERS = 28
mock_teacher = make_mock_teacher(
    hidden=TEACHER_HIDDEN,
    num_layers=TEACHER_TRANSFORMER_LAYERS,
    batch=BATCH_SIZE,
    seq=SEQ_LEN,
    device=DEVICE,
)
print(f"  OK: mock teacher ready ({TEACHER_TRANSFORMER_LAYERS} layers, hidden={TEACHER_HIDDEN})")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Initialize student components
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Initializing student components...")

print("  Loading latentBERT (DUS, 40 layers from ModernBERT-large)...")
latent_bert = create_latentbert(
    model_id="answerdotai/ModernBERT-large",
    target_layers=STUDENT_LAYERS
)
latent_bert = latent_bert.to(DEVICE)
latent_bert.eval()
param_count = sum(p.numel() for p in latent_bert.parameters())
print(f"  OK: latentBERT ready -- {param_count:,} parameters")

input_projector = InputProjector().to(DEVICE)
print(f"  OK: InputProjector({TEACHER_HIDDEN} -> 2048 -> {STUDENT_HIDDEN})")

feature_projectors = nn.ModuleDict({
    "20": FeatureProjector(component_id="feat_proj_20").to(DEVICE),
    "30": FeatureProjector(component_id="feat_proj_30").to(DEVICE),
    "40": FeatureProjector(component_id="feat_proj_40").to(DEVICE),
})
print(f"  OK: FeatureProjectors x3 ({STUDENT_HIDDEN} -> {TEACHER_HIDDEN})")

layer_mapping = {20: 14, 30: 21, 40: 28}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Forward pass
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/4] Running forward pass...")

# Dummy input tokens
input_ids = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long, device=DEVICE)
attention_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long, device=DEVICE)

# ---- Teacher forward (mock) ----
with torch.no_grad():
    teacher_outputs = mock_teacher(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    teacher_embeddings = teacher_outputs.hidden_states[0].to(DEVICE)
    teacher_targets = {
        s_idx: teacher_outputs.hidden_states[t_idx].to(DEVICE)
        for s_idx, t_idx in layer_mapping.items()
    }

print(f"  Teacher embeddings shape : {teacher_embeddings.shape}")
assert not torch.isnan(teacher_embeddings).any(), "NaN detected in teacher embeddings!"
for s_idx, t in teacher_targets.items():
    assert not torch.isnan(t).any(), f"NaN detected in teacher target state for layer {s_idx}!"

# ---- InputProjector ----
with torch.no_grad():
    student_inputs_embeds = input_projector(teacher_embeddings)

print(f"  After InputProjector     : {student_inputs_embeds.shape}")
assert not torch.isnan(student_inputs_embeds).any(), "NaN detected after InputProjector!"

# ---- Student (latentBERT) forward ----
with torch.no_grad():
    student_outputs = latent_bert(
        inputs_embeds=student_inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

num_hidden = len(student_outputs.hidden_states)
print(f"  Student hidden states    : {num_hidden} (expected {STUDENT_LAYERS + 1})")
for i, h in enumerate(student_outputs.hidden_states):
    assert not torch.isnan(h).any(), f"NaN detected in student hidden state at index {i}!"

# ---- Student Logits (Tokens) ----
logits = student_outputs.logits
print(f"  Student logits shape     : {logits.shape}")
assert not torch.isnan(logits).any(), "NaN detected in student logits (output tokens)!"

# ---- FeatureProjectors ----
print("\n  FeatureProjectors:")
with torch.no_grad():
    for s_idx, t_idx in layer_mapping.items():
        h_state = student_outputs.hidden_states[s_idx]
        projected = feature_projectors[str(s_idx)](h_state)
        target = teacher_targets[s_idx]

        print(f"    Layer {s_idx:2d}: student={tuple(h_state.shape)} "
              f"-> projected={tuple(projected.shape)} "
              f"| teacher target={tuple(target.shape)}")

        assert not torch.isnan(projected).any(), f"NaN detected in projected student state for layer {s_idx}!"

        assert h_state.shape   == (BATCH_SIZE, SEQ_LEN, STUDENT_HIDDEN), \
            f"Student hidden_state[{s_idx}]: expected (B,T,{STUDENT_HIDDEN}), got {h_state.shape}"
        assert projected.shape == (BATCH_SIZE, SEQ_LEN, TEACHER_HIDDEN), \
            f"Projected[{s_idx}]: expected (B,T,{TEACHER_HIDDEN}), got {projected.shape}"
        assert target.shape    == (BATCH_SIZE, SEQ_LEN, TEACHER_HIDDEN), \
            f"Teacher target[{s_idx}]: expected (B,T,{TEACHER_HIDDEN}), got {target.shape}"
        assert projected.shape == target.shape, \
            f"Projected and target shapes do not match at layer {s_idx}"

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SMOKE TEST PASSED")
print("  All tensor shapes are correct.")
print("  Pipeline InputProjector -> latentBERT -> FeatureProjectors works.")
print("=" * 60)
