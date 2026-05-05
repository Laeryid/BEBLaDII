import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from .base import BEComponent
from .dus import DUSModel
from .projectors import InputProjector, FeatureProjector

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False

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
        
        # Загружаем учителя в bfloat16 (без 4-битного квантования)
        self.teacher = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map if not XLA_AVAILABLE else None,
            trust_remote_code=True,
            attn_implementation="sdpa"
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
        if student_device is None:
            if XLA_AVAILABLE:
                student_device = xm.xla_device()
            else:
                student_device = "cuda" if torch.cuda.is_available() else "cpu"
                
        self.student_device = student_device
        print(f"Student device: {self.student_device}")
        
        if XLA_AVAILABLE:
            # На TPU мы переносим и учителя, и ученика на XLA устройство
            self.teacher.to(self.student_device)
            
        self.student.to(self.student_device)
        self.input_projector.to(self.student_device)
        self.feature_projectors.to(self.student_device)
        
        # Переводим обучаемые компоненты в bfloat16
        self.student.to(torch.bfloat16)
        self.input_projector.to(torch.bfloat16)
        self.feature_projectors.to(torch.bfloat16)
        print("DEBUG: ReasoningDistiller initialized with bfloat16 weights.")
        
        # Настройка маппинга слоев (Student -> Teacher)
        self.layer_mapping = {
            20: 14, # Middle
            30: 21, # 3/4
            40: 27  # Last (Pre-norm)
        }

    def _check_nan(self, tensor, name):
        """Вспомогательная функция для отладки NaN/Inf."""
        if torch.isnan(tensor).any():
            print(f"!!! DEBUG: NaN detected in {name}")
            return True
        if torch.isinf(tensor).any():
            print(f"!!! DEBUG: Inf detected in {name}")
            return True
        return False

    def forward(self, input_ids, attention_mask=None):
        """
        Прямой проход для дистилляции с отладкой.
        """
        # 1. Проход Teacher
        # Определяем устройство учителя (обычно первая видеокарта в device_map)
        try:
            teacher_device = next(self.teacher.parameters()).device
        except StopIteration:
            # XLA FSDP может скрывать параметры под FlatParameter wrapper'ами
            teacher_device = getattr(self, "student_device", input_ids.device)
            
        t_input_ids = input_ids.to(teacher_device)
        t_attention_mask = attention_mask.to(teacher_device) if attention_mask is not None else None
        
        # ГАРАНТИРУЕМ, что учитель в eval режиме (FSDP может переключить его в train рекурсивно)
        self.teacher.eval()

        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=t_input_ids,
                attention_mask=t_attention_mask,
                output_hidden_states=True
            )
            teacher_embeddings = teacher_outputs.hidden_states[0].to(self.student_device).to(torch.bfloat16)
            # self._check_nan(teacher_embeddings, "Teacher Embeddings")
            
            teacher_targets = {
                s_idx: teacher_outputs.hidden_states[t_idx].to(self.student_device).to(torch.bfloat16)
                for s_idx, t_idx in self.layer_mapping.items()
            }
            for idx, t in teacher_targets.items():
                pass # self._check_nan(t, f"Teacher Target Layer {idx}")
            
            # [DEBUG] Нормы активаций на первом шаге
            if getattr(self, "_debug_first_step", True) and XLA_AVAILABLE and xm.get_local_ordinal() == 0:
                print(f"--- [DEBUG ACTIVATIONS] ---")
                print(f"Teacher Embeds Norm: {torch.norm(teacher_embeddings.float()).item():.2f}")
                first_t_idx = list(self.layer_mapping.keys())[0]
                print(f"Teacher Target L{first_t_idx} Norm: {torch.norm(teacher_targets[first_t_idx].float()).item():.2f}")
            
        # 2. Подготовка входа для Student
        student_inputs_embeds, mu, logvar = self.input_projector(teacher_embeddings)
        # self._check_nan(student_inputs_embeds, "InputProjector Output")
        
        # ВНИМАНИЕ: attention_mask может приходить от данных Учителя (с cuda:0), поэтому переводим его на student_device.
        student_attention_mask = attention_mask.to(self.student_device) if attention_mask is not None else None
        
        student_outputs = self.student(
            inputs_embeds=student_inputs_embeds,
            attention_mask=student_attention_mask,
            output_hidden_states=True
        )
        # self._check_nan(student_outputs.hidden_states[-1], "Student Final Hidden State")
        
        # 4. Проецирование состояний ученика обратно в пространство Qwen
        projected_student_states = {}
        for idx, h_state in {idx: student_outputs.hidden_states[idx] for idx in self.layer_mapping.keys()}.items():
            proj = self.feature_projectors[str(idx)](h_state)
            # self._check_nan(proj, f"FeatureProjector {idx} Output")
            projected_student_states[idx] = proj

        if getattr(self, "_debug_first_step", True) and XLA_AVAILABLE and xm.get_local_ordinal() == 0:
            print(f"Student Input Projector Norm: {torch.norm(student_inputs_embeds.float()).item():.2f}")
            first_p_idx = list(self.layer_mapping.keys())[0]
            print(f"Student Projector L{first_p_idx} Output Norm: {torch.norm(projected_student_states[first_p_idx].float()).item():.2f}")
            self._debug_first_step = False

        return projected_student_states, teacher_targets, mu, logvar

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        """Переопределяем state_dict, чтобы ИСКЛЮЧИТЬ веса учителя из сохранения."""
        full_dict = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        # Фильтруем ключи, не трогая те, что относятся к учителю
        filtered_dict = {k: v for k, v in full_dict.items() if not k.startswith(prefix + 'teacher.')}
        return filtered_dict

if __name__ == "__main__":
    # Тест инициализации (потребует много памяти)
    # distill = ReasoningDistiller()
    pass
