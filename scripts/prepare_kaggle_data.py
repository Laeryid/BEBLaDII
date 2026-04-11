import os
import shutil
import json
from pathlib import Path

# Попробуем оба варианта импорта для надежности
try:
    from indexed_parquet import IndexedParquetDataset
except ImportError:
    from indexed_parquet_dataset import IndexedParquetDataset

def prepare_kaggle_package():
    # 1. Пути
    root = Path(".")
    staging = root / "kaggle_upload"
    staging_data = staging / "data"
    
    # 2. Очистка старой папки
    if staging.exists():
        shutil.rmtree(staging)
    staging_data.mkdir(parents=True)
    
    # 3. Копирование весов
    weights_src = root / "storage" / "components" / "model" / "latentBERT" / "v1.0" / "weights.pt"
    if weights_src.exists():
        print(f"Копирование весов: {weights_src}...")
        shutil.copy2(weights_src, staging / "weights.pt")
    else:
        print(f"WARN: Веса не найдены по адресу {weights_src}")

    # 4. Клонирование датасетов
    # Stage 1: Awakening (90k CulturaX, 10k Magpie)
    # Stage 2: Reasoning (50k Magpie, 50k OpenThoughts, 30k CulturaX)
    
    data_plans = [
        {"name": "CulturaX_cs_Awakening", "target": "Awakening","path": "data/CulturaX", "pattern": "cs_part_00000.parquet", "count": 45000},
        {"name": "CulturaX_ru_Awakening", "target": "Awakening","path": "data/CulturaX", "pattern": "ru_part_00000.parquet", "count": 45000},
        {"name": "CulturaX_cs_Reasoning", "target": "Reasoning","path": "data/CulturaX", "pattern": "cs_part_00000.parquet", "count": 15000},
        {"name": "CulturaX_ru_Reasoning", "target": "Reasoning","path": "data/CulturaX", "pattern": "ru_part_00000.parquet", "count": 15000},
        {"name": "magpie_awakening", "target": "Awakening","path": "data/magpie_reasoning", "count": 10000},
        {"name": "magpie_reasoning", "target": "Reasoning","path": "data/magpie_reasoning", "count": 50000},
        {"name": "open_thoughts", "target": "Reasoning","path": "data/open_thoughts", "count": 50000}
    ]
    
    for plan in data_plans:
        src_dir = root / plan["path"]
        if not src_dir.exists():
            print(f"Пропуск {plan['name']}: путь {src_dir} не найден.")
            continue
            
        # 4.1 Формируем целевой путь: data/{Target}/{Name}
        dest_path = staging_data / plan["target"] / plan["name"]
        print(f"\n--- Обработка: {plan['name']} ({plan['target']}) ---")
        
        # 4.2 Загрузка датасета (с фильтрацией по паттерну, если есть)
        if "pattern" in plan:
            pattern = plan["pattern"]
            matched_files = list(src_dir.glob(pattern))
            if not matched_files:
                print(f"WARN: Файлы по паттерну {pattern} не найдены в {src_dir}")
                continue
            print(f"Найдено файлов по паттерну: {len(matched_files)}")
            # Предполагаем, что библиотека умеет принимать список файлов
            # Если нет — IndexedParquetDataset.from_folder обычно работает в корне
            ds = IndexedParquetDataset.from_folder(src_dir, pattern=pattern)
        else:
            ds = IndexedParquetDataset.from_folder(src_dir)
            
        # 4.3 Сэмплирование
        if len(ds) > plan["count"]:
            print(f"Сэмплирование: {len(ds)} -> {plan['count']} строк")
            ds = ds.sample(n=plan["count"])
        else:
            print(f"Используем все доступные строки: {len(ds)}")
            
        # 4.4 Клонирование
        if dest_path.suffix != ".parquet":
            dest_path = dest_path.with_suffix(".parquet")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
            
        print(f"Клонирование в {dest_path}...")
        ds.clone(dest_path)

    # 5. Создание метаданных Kaggle
    metadata = {
        "title": "BEBLaDII-Phase1-Resources",
        "id": "bogdanbuliakov/bebladii-resources",
        "licenses": [{"name": "CC0-1.0"}]
    }
    
    with open(staging / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("\nПодготовка завершена!")
    print(f"Все файлы в: {staging.absolute()}")
    print("Теперь вы можете запустить 'kaggle datasets create -p kaggle_upload'")

if __name__ == "__main__":
    prepare_kaggle_package()
