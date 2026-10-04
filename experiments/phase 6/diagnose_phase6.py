import os, sys, math, argparse
import torch
import torch.nn.functional as F
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = "C:/Experiments/BEBLaDII"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "phase 6"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from transformers import AutoTokenizer
from beb_la_dii.utils.loss import safe_normalize
from evaluate_phase6_metrics import BEBLaDIIPhase6Eval

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default=os.path.join(PROJECT_ROOT, "experiments", "phase 6", "checkpoints", "planB_phase6_checkpoints_phase6_ca_layers_step_6000.pth"))
p.add_argument("--n", type=int, default=6)
p.add_argument("--T", type=int, default=96)
args = p.parse_args()
torch.manual_seed(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    'qwen_path': "Qwen/Qwen2.5-1.5B", 'modernbert_path': "answerdotai/ModernBERT-large",
    'encoder_path': os.path.join(PROJECT_ROOT, "experiments", "phase 1", "planB_phase1_checkpoints_phase1_vae_step_20000.pth"),
    'phase4_path': os.path.join(PROJECT_ROOT, "experiments", "phase 4", "local_checkpoints", "phase4_step_85995.pth"),
    'sep_token_path': os.path.join(PROJECT_ROOT, "storage", "components", "sep_token.pt"),
    'phase6_ckpt': None,
}
tok = AutoTokenizer.from_pretrained(config['qwen_path'])
model = BEBLaDIIPhase6Eval(config).to(dev)
latent_dict = torch.load(os.path.join(PROJECT_ROOT, "experiments", "phase 5", "local", "latent_dict.pt"), map_location=dev).float()
void = torch.load(os.path.join(PROJECT_ROOT, "storage", "components", "void_token.pt"), map_location=dev).float()
void = safe_normalize(void, dim=-1)

# --- состояния: orig (Phase4) / full (весь ckpt) ---
st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
sd = model.state_dict()
dus_keys = [k for k in st if "ca_layer" not in k]
orig_dus = {k: sd[k].clone() for k in dus_keys if k in sd}
print(f"ckpt keys: {len(st)}, non-CA DUS keys: {len(dus_keys)}, matched in model: {len(orig_dus)}")
missing = [k for k in dus_keys if k not in sd]
print("unmatched non-CA keys:", missing[:5])
r = model.load_state_dict(st, strict=False)
full_dus = {k: model.state_dict()[k].clone() for k in orig_dus}
ca_modules = list(model.ca_layers.values())
saved_out = [(c.out_proj.weight.data.clone(), c.out_proj_sa.weight.data.clone()) for c in ca_modules]

def set_variant(v):
    sd_now = model.state_dict()
    src = orig_dus if v == "caonly" else full_dus
    for k, t in src.items(): sd_now[k].copy_(t)
    for c, (a, b) in zip(ca_modules, saved_out):
        if v == "noCA":
            c.out_proj.weight.data.zero_(); c.out_proj_sa.weight.data.zero_()
        else:
            c.out_proj.weight.data.copy_(a); c.out_proj_sa.weight.data.copy_(b)

# --- данные ---
df = pd.read_parquet(os.path.join(PROJECT_ROOT, "BEBLaDII-planB-Phase6-Data", "phase 6", "data", "train_phase6.parquet"))
df = df.sample(200, random_state=1)
rows = []
for _, r_ in df.iterrows():
    if len(tok(str(r_['A'])).input_ids) >= 24: rows.append(r_)
    if len(rows) == args.n: break
T = args.T
with torch.no_grad():
    Zp, Zt, Mq, ids_a, Ma = [], [], [], [], []
    Lq = 64
    for r_ in rows:
        q = tok(str(r_['Q']), truncation=True, max_length=Lq, padding='max_length', return_tensors='pt').to(dev)
        a = tok(str(r_['A']), truncation=True, max_length=T, padding='max_length', return_tensors='pt').to(dev)
        z_q, _, _ = model.encoder(model.qwen_embeddings(q.input_ids)); Zp.append(safe_normalize(z_q.float(), dim=-1))
        z_a, _, _ = model.encoder(model.qwen_embeddings(a.input_ids)); z_a = safe_normalize(z_a.float(), dim=-1)
        m = a.attention_mask
        z_a = torch.where(m.unsqueeze(-1).bool(), z_a, void.view(1, 1, -1))
        Zt.append(z_a); Mq.append(q.attention_mask); ids_a.append(a.input_ids); Ma.append(m)
    Zp = torch.cat(Zp); Zt = torch.cat(Zt); Mq = torch.cat(Mq); ids_a = torch.cat(ids_a); Ma = torch.cat(Ma).bool()
B = Zt.shape[0]

def noise(x0, t):
    eps = safe_normalize(torch.randn_like(x0), dim=-1)
    t = t.unsqueeze(-1)
    return safe_normalize(torch.cos(t * math.pi / 2) * x0 + torch.sin(t * math.pi / 2) * eps, dim=-1)

def run(t_actual):
    zn = noise(Zt, t_actual)
    raw, _ = torch.matmul(zn, latent_dict.T).max(dim=-1)
    t_rep = (1 - raw).clamp(0, 1)
    h39, zp = model.forward_step(zn, Zp, Mq, t_actual, t_rep)
    return zn, h39, zp

def metrics(zn, h39, zp, mask, tag):
    c = F.cosine_similarity(zp, Zt, dim=-1)[mask].mean().item()
    b = F.cosine_similarity(zn, Zt, dim=-1)[mask].mean().item()
    ch = F.cosine_similarity(h39, Zt, dim=-1)[mask].mean().item()
    top1 = (torch.matmul(zp[mask], latent_dict.T).argmax(-1) == ids_a[mask]).float().mean().item()
    top1h = (torch.matmul(h39[mask], latent_dict.T).argmax(-1) == ids_a[mask]).float().mean().item()
    print(f"{tag:40s} cos_pred={c:.3f} cos_h39={ch:.3f} cos_noisy(baseline)={b:.3f} top1_pred={top1:.3f} top1_h39={top1h:.3f}")

with torch.no_grad():
    for v in ["full", "caonly", "noCA"]:
        set_variant(v)
        print(f"\n===== variant: {v} (content tokens only) =====")
        print("-- uniform t (eval-style: t_global=t) --")
        for t in [0.2, 0.5, 0.8, 1.0]:
            torch.manual_seed(1)
            zn, h39, zp = run(torch.full((B, T), t, device=dev))
            metrics(zn, h39, zp, Ma, f"uniform t={t}")
        print("-- per-token random t (train-style, binned by token t) --")
        torch.manual_seed(2)
        tt = torch.randint(1, 26, (B, T), device=dev) / 25.0
        zn, h39, zp = run(tt)
        for lo, hi in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]:
            mk = Ma & (tt > lo) & (tt <= hi)
            if mk.sum() > 0: metrics(zn, h39, zp, mk, f"mixed t in ({lo},{min(hi,1.0)}]")
        vm = ~Ma
        cv = F.cosine_similarity(zp, Zt, dim=-1)[vm].mean().item()
        print(f"{'mixed: void tokens cos_pred':40s} {cv:.3f}")
