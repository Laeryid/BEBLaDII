import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from .base import BEComponent
from .dus import DUSModel
from .projectors import InputProjector, FeatureProjector

class ReasoningDistiller(nn.Module):
    """
    Класс-обертка для дистилляции ModernBERT из DeepSeek-R1-7B.
    Поддерживает работу с компонентами BEBLaDII.
    """
    def __init__(self, 
                 teacher_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                 student=None,
                 input_projector=None,
                 feature_projectors=None,
                 device_map="auto",
                 student_device=None):
        super().__init__()
        
        # 1. Loading Teacher (frozen, 4-bit)
        import os
        
        # Определение путей загрузки (Приоритет: Локально -> Kaggle Models -> Kaggle Dataset -> HF ID)
        short_name = "deepseek-7b" if "DeepSeek-R1-Distill-Qwen-7B" in teacher_id else teacher_id.split("/")[-1]
        
        potential_paths = [
            os.path.join("storage", "prebuilt", short_name), # Local Dev
            "/kaggle/input/models/deepseek-ai/deepseek-r1/transformers/deepseek-r1-distill-qwen-7b/2", # Kaggle Models (Primary)
            os.path.join("/kaggle/input/bebladii-resources/prebuilt", short_name), # Fallback Dataset
        ]
        
        load_path = teacher_id
        for path in potential_paths:
            if os.path.exists(path):
                load_path = path
                break
        
        print(f"Loading Teacher from: {load_path}...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.teacher = AutoModelForCausalLM.from_pretrained(
            load_path,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # 2. Использование переданных компонентов или инициализация по умолчанию
        self.student = student or DUSModel()
        self.input_projector = input_projector or InputProjector()
        
        if feature_projectors:
            self.feature_projectors = feature_projectors
        else:
            self.feature_projectors = nn.ModuleDict({
                "20": FeatureProjector(component_id="feat_proj_20"),
                "30": FeatureProjector(component_id="feat_proj_30"),
                "40": FeatureProjector(component_id="feat_proj_40")
            })
        
        # 3. Явный перенос обучаемых компонентов на нужный device.
        # Мы НЕ вызываем .to() на всём дистилляторе, т.к. учитель использует
        # device_map='auto' (bitsandbytes), и его трогать нельзя.
        # student_device по умолчанию - 'cuda', если доступна, иначе 'cpu'.
        if student_device is None:
            student_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.student_device = student_device
        print(f"Student device: {self.student_device}")
        
        self.student.to(self.student_device).half()
        self.input_projector.to(self.student_device).half()
        self.feature_projectors.to(self.student_device).half()
        
        # Настройка маппинга слоев (Student -> Teacher)
        self.layer_mapping = {
            20: 14, # Middle
            30: 21, # 3/4
            40: 28  # Last
        }

    def forward(self, input_ids, attention_mask=None):
        """
        Прямой проход для дистилляции.
        """
        # 1. Проход Teacher
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # Извлекаем эмбеддинги (первый слой) и нужные скрытые состояния.
            # Явно переносим на student_device: с device_map='auto' учитель
            # может раскладывать слои по разным GPU, и последний слой
            # может оказаться не там, где student.
            teacher_embeddings = teacher_outputs.hidden_states[0].to(self.student_device)
            teacher_targets = {
                s_idx: teacher_outputs.hidden_states[t_idx].to(self.student_device)
                for s_idx, t_idx in self.layer_mapping.items()
            }
            
        # 2. Подготовка входа для Student
        # Эмбеддинги учителя -> Проектор -> Вход ученика
        student_inputs_embeds = self.input_projector(teacher_embeddings)
        
        # 3. Проход Student
        student_outputs = self.student(
            inputs_embeds=student_inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Извлекаем скрытые состояния ученика (индексы слоев 1-indexed для конфига)
        # Но в transformers hidden_states[0] - это входной эмбеддинг, hidden_states[1] - 1-й слой.
        # Поэтому hidden_states[20] - это слой 20.
        student_samples = {
            idx: student_outputs.hidden_states[idx] 
            for idx in self.layer_mapping.keys()
        }
        
        # 4. Проецирование состояний ученика обратно в пространство Qwen
        projected_student_states = {}
        for idx, h_state in student_samples.items():
            projected_student_states[idx] = self.feature_projectors[str(idx)](h_state)
            
        return projected_student_states, teacher_targets

if __name__ == "__main__":
    # Тест инициализации (потребует много памяти)
    # distill = ReasoningDistiller()
    pass
