import torch
import gc
import os

def emergency_save():
    """
    Принудительное сохранение весов модели из памяти ядра Kaggle.
    Полезно, если обучение зависло, сессия истекает или стандартные методы сохранения не сработали.
    """
    print("Попытка принудительного сохранения из памяти...")

    found = False
    # Проверяем стандартные имена переменных в глобальной области видимости
    candidates = ['distiller', 'model', 'distill_model']

    for name in candidates:
        if name in globals():
            obj = globals()[name]
            print(f"Найдена переменная '{name}'. Сохраняю...")
            try:
                # Если это наш дистиллер (ReasoningDistiller), сохраняем стейт-дикт его ключевых частей
                if hasattr(obj, 'student'):
                    state = {
                        "latentBERT_state_dict": obj.student.state_dict(),
                        "input_projector": obj.input_projector.state_dict(),
                        "feature_projectors": obj.feature_projectors.state_dict(),
                        "step_emergency": getattr(obj, 'current_step', 'unknown') 
                    }
                else:
                    # Для обычных моделей сохраняем весь state_dict
                    state = obj.state_dict()
                    
                save_path = '/kaggle/working/EMERGENCY_WEIGHTS_FINAL.pt'
                torch.save(state, save_path)
                print(f"!!! УСПЕХ: Веса сохранены в {save_path}")
                print(f"Размер файла: {os.path.getsize(save_path) // 1024**2} MB")
                found = True
                break
            except Exception as e:
                print(f"Ошибка при сохранении {name}: {e}")

    if not found:
        print("В глобальных переменных модель не найдена. Пробую глубокий поиск через gc...")
        for obj in gc.get_objects():
            try:
                # Ищем объект, у которого есть характерные поля нашего ReasoningDistiller
                if hasattr(obj, 'student') and hasattr(obj, 'feature_projectors'):
                    print("Найден подходящий объект в памяти через gc. Сохраняю...")
                    state = {
                        "latentBERT_state_dict": obj.student.state_dict(),
                        "input_projector": obj.input_projector.state_dict(),
                        "feature_projectors": obj.feature_projectors.state_dict(),
                    }
                    torch.save(state, '/kaggle/working/GC_RESCUED_WEIGHTS.pt')
                    print(f"!!! УСПЕХ: Веса спасены через gc в /kaggle/working/GC_RESCUED_WEIGHTS.pt")
                    found = True
                    break
            except:
                continue

    if found:
        from IPython.display import FileLink
        # Попытка вывести ссылку на скачивание (работает в Jupyter/Kaggle Notebooks)
        print("\nСКАЧАЙТЕ ФАЙЛ СЕЙЧАС:")
        filename = 'EMERGENCY_WEIGHTS_FINAL.pt' if os.path.exists('/kaggle/working/EMERGENCY_WEIGHTS_FINAL.pt') else 'GC_RESCUED_WEIGHTS.pt'
        display(FileLink(filename))
    else:
        print("КРИТИЧЕСКАЯ ОШИБКА: Модель не найдена в памяти ядра.")

if __name__ == "__main__":
    emergency_save()
