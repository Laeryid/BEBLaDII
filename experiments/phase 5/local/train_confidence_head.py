import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm

PROJECT_ROOT = "C:/Experiments/BEBLaDII"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.beb_la_dii.model.confidence_head import ConfidenceHead

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Пути
    dataset_path = os.path.join(PROJECT_ROOT, "experiments", "phase 5", "local", "unified_real_dataset.pt")
    model_save_path = os.path.join(PROJECT_ROOT, "experiments", "phase 5", "local", "confidence_head_v2.pt")

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print("Loading dataset...")
    dataset_dict = torch.load(dataset_path, map_location="cpu")
    
    h39_raw = dataset_dict["h39_raw"]
    dus_final = dataset_dict["dus_final"]
    t_reported = dataset_dict["t_reported"]
    labels = dataset_dict["labels"]

    print(f"h39_raw shape: {h39_raw.shape}")
    print(f"dus_final shape: {dus_final.shape}")
    print(f"t_reported shape: {t_reported.shape}")
    print(f"labels shape: {labels.shape}")

    # Создаем датасет
    full_dataset = TensorDataset(h39_raw, dus_final, t_reported, labels)
    
    # Train / Val Split (90/10)
    total_size = len(full_dataset)
    val_size = int(total_size * 0.1)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Initializing ConfidenceHead...")
    # Инициализация новой головы с кастомным локальным вниманием
    model = ConfidenceHead(
        h39_dim=1024,
        attn_dim=256,
        num_heads=4,
        window_size=32,
        conf_dim=6,
        mlp_hidden=512
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    epochs = 10

    # Будем использовать MSELoss, так как наши сигналы - это непрерывные величины [0, 1]
    criterion = nn.MSELoss()

    signal_names = ["DictProximity", "ModelConf", "Coherence", "Complexity", "Cryst", "Stuck"]

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_channel_losses = torch.zeros(6, device=device)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_h39_raw, batch_dus_final, batch_t, batch_labels in pbar:
            batch_h39_raw = batch_h39_raw.to(device)
            batch_dus_final = batch_dus_final.to(device)
            batch_t = batch_t.to(device)
            batch_labels = batch_labels.to(device)

            B, T, _ = batch_h39_raw.shape

            # Инициализируем conf_prev нулями, так как это независимые кадры
            conf_prev = torch.zeros((B, T, 6), device=device)
            # Нет паддинга, все токены валидны
            attn_mask = torch.ones((B, T), device=device)

            optimizer.zero_grad()
            preds = model(batch_h39_raw, batch_dus_final, conf_prev, batch_t, attention_mask=attn_mask)
            
            # Loss
            loss = criterion(preds, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            # Считаем лосс по каждому каналу отдельно для аналитики
            with torch.no_grad():
                mse_per_channel = F.mse_loss(preds, batch_labels, reduction='none').mean(dim=[0, 1])
                train_channel_losses += mse_per_channel

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader)
        train_channel_losses /= len(train_loader)

        # Валидация
        model.eval()
        val_loss = 0.0
        val_channel_losses = torch.zeros(6, device=device)
        
        with torch.no_grad():
            for batch_h39_raw, batch_dus_final, batch_t, batch_labels in val_loader:
                batch_h39_raw = batch_h39_raw.to(device)
                batch_dus_final = batch_dus_final.to(device)
                batch_t = batch_t.to(device)
                batch_labels = batch_labels.to(device)
                B, T, _ = batch_h39_raw.shape

                conf_prev = torch.zeros((B, T, 6), device=device)
                attn_mask = torch.ones((B, T), device=device)

                preds = model(batch_h39_raw, batch_dus_final, conf_prev, batch_t, attention_mask=attn_mask)
                val_loss += criterion(preds, batch_labels).item()
                val_channel_losses += F.mse_loss(preds, batch_labels, reduction='none').mean(dim=[0, 1])

        val_loss /= len(val_loader)
        val_channel_losses /= len(val_loader)

        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Красивый вывод лоссов по каждому сигналу
        ch_strs = []
        for i, name in enumerate(signal_names):
            ch_strs.append(f"{name}: {val_channel_losses[i]:.4f}")
        print("Val Channel MSE: " + " | ".join(ch_strs))
        print("-" * 50)

    print(f"Saving model to {model_save_path}...")
    torch.save(model.state_dict(), model_save_path)
    print("Training complete!")

if __name__ == "__main__":
    train()
