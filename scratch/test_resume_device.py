import sys
import torch

sys.path.append("experiments/phase 1")
from train_phase1 import (
    GrassmannianPhase1Model,
    create_hybrid_optimizers,
    load_checkpoint,
    EnvironmentConfig,
)

def test_resume_device():
    env = EnvironmentConfig.detect()
    device = torch.device("cpu")

    model = GrassmannianPhase1Model(vocab_size=131072)
    opt_muon, opt_adamw = create_hybrid_optimizers(model)

    ckpt_path = "experiments/phase 1/checkpoints/phase1_best_model.pt"
    start_step, _ = load_checkpoint(
        path=ckpt_path,
        model=model,
        opt_muon=opt_muon,
        opt_adamw=opt_adamw,
        device=device,
    )
    print(f"Resumed at step: {start_step}")

    # Проверяем, что все тензоры состояния Muon лежат на device
    for p, state in opt_muon.state.items():
        if "momentum_buffer" in state:
            assert state["momentum_buffer"].device == device, f"Buffer on wrong device: {state['momentum_buffer'].device}"

    # Создаем фиктивные градиенты и делаем opt_muon.step()
    for p in model.parameters():
        if p.ndim == 2:
            p.grad = torch.randn_like(p)

    opt_muon.step()
    print("opt_muon.step() completed successfully without device errors!")

if __name__ == "__main__":
    test_resume_device()
