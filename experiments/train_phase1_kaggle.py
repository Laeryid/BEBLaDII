import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# Импорты проекта
from src.beb_la_dii.model.distiller import ReasoningDistiller
from src.beb_la_dii.utils.tokenizer import get_tokenizer
from src.beb_la_dii.utils.loss import DistillationLoss
from src.beb_la_dii.utils.data import DistillationDataset

# Константы
MODEL_NAME = "answerdotai/ModernBERT-large"
TEACHER_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_LENGTH = 512
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
EPOCHS = 1
LEARNING_RATE = 5e-5
LIMIT_SAMPLES = 100000

def train():
    # 1. WandB Log In
    if os.environ.get("WANDB_API_KEY"):
        wandb.init(project="beb-la-dii-phase1", name="latentbert-distillation")
        
    # 2. Инициализация токенизатора
    tokenizer = get_tokenizer(TEACHER_NAME)
    
    # 3. Подготовка данных
    dataset = DistillationDataset(tokenizer, max_length=MAX_LENGTH, limit=LIMIT_SAMPLES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 4. Инициализация моделей
    device = "cuda" if torch.cuda.is_available() else "cpu"
    distiller = ReasoningDistiller(TEACHER_NAME, MODEL_NAME, device_map="auto")
    # Переводим только обучаемые части на GPU если нужно (дистиллятор сам распределит)
    # Но для обучения нам нужно включить grads только для student и проекторов
    for param in distiller.student.parameters():
        param.requires_grad = True
    for param in distiller.input_projector.parameters():
        param.requires_grad = True
    for param in distiller.feature_projectors.parameters():
        param.requires_grad = True
        
    # 5. Оптимизатор и Лосс
    optimizer = AdamW(filter(lambda p: p.requires_grad, distiller.parameters()), lr=LEARNING_RATE)
    criterion = DistillationLoss()
    
    # 6. Тренировочный цикл
    distiller.train()
    progress_bar = tqdm(dataloader, desc="Training Phase 1")
    
    accum_loss = 0.0
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Форвард
        student_states, teacher_targets = distiller(input_ids, attention_mask)
        
        # Расчет лосса
        loss = criterion(student_states, teacher_targets)
        loss = loss / GRAD_ACCUM_STEPS # Нормализация для накопления градиента
        
        # Бэквард
        loss.backward()
        accum_loss += loss.item()
        
        # Шаг оптимизатора
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            # Логирование
            avg_loss = accum_loss * GRAD_ACCUM_STEPS / GRAD_ACCUM_STEPS
            progress_bar.set_postfix({"loss": avg_loss})
            if wandb.run:
                wandb.log({"loss": avg_loss})
            accum_loss = 0.0
            
    # 7. Сохранение результатов
    print("Сохранение модели...")
    torch.save({
        "student_state_dict": distiller.student.state_dict(),
        "input_projector": distiller.input_projector.state_dict(),
        "feature_projectors": distiller.feature_projectors.state_dict()
    }, "latentbert_phase1_distilled.pt")
    
    print("Дистилляция Фазы 1 завершена.")

if __name__ == "__main__":
    train()
