import os
import shutil
import json
from pathlib import Path

# Попробуем оба варианта импорта для надежности
try:
    from indexed_parquet import IndexedParquetDataset
except ImportError:
    from indexed_parquet_dataset import IndexedParquetDataset

import sys
import codecs

# Fix for Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Fallback for older python
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

def prepare_kaggle_package():
    # 1. Paths
    root = Path(".")
    staging = root / "kaggle_upload"
    staging_data = staging / "data"
    
    # 2. Cleanup
    if staging.exists():
        print(f"Cleaning up {staging}...")
        shutil.rmtree(staging)
    staging_data.mkdir(parents=True)
    
    # 3. Mirroring components and prebuilts (Student & Teacher)
    mirror_paths = [
        ("storage/components/model/latentBERT/v1.0", "components/model/latentBERT/v1.0"),
        ("storage/prebuilt/latentBERT/v1.0", "prebuilt/latentBERT/v1.0"),
        ("storage/prebuilt/deepseek-7b", "prebuilt/deepseek-7b"),
    ]
    
    for src_rel, dst_rel in mirror_paths:
        src = root / src_rel
        dst = staging / dst_rel
        if src.exists():
            print(f"Mirroring: {src_rel} -> {dst_rel}...")
            if dst.exists(): shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst)
        else:
            print(f"WARN: Path {src_rel} not found. Skipping.")

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
            print(f"Skipping {plan['name']}: {src_dir} not found.")
            continue
            
        # 4.1 Set destination path: data/{Target}/{Name}
        dest_path = staging_data / plan["target"] / plan["name"]
        print(f"\n--- Processing: {plan['name']} ({plan['target']}) ---")
        
        # 4.2 Загрузка датасета (с фильтрацией по паттерну, если есть)
        if "pattern" in plan:
            pattern = plan["pattern"]
            matched_files = list(src_dir.glob(pattern))
            if not matched_files:
                print(f"WARN: Файлы по паттерну {pattern} не найдены в {src_dir}")
                continue
            print(f"Found files by pattern: {len(matched_files)}")
            # Предполагаем, что библиотека умеет принимать список файлов
            # Если нет — IndexedParquetDataset.from_folder обычно работает в корне
            ds = IndexedParquetDataset.from_folder(src_dir, pattern=pattern)
        else:
            ds = IndexedParquetDataset.from_folder(src_dir)
            
        # 4.3 Сэмплирование
        if len(ds) > plan["count"]:
            print(f"Sampling: {len(ds)} -> {plan['count']} rows")
            ds = ds.sample(n=plan["count"])
        else:
            print(f"Using all available rows: {len(ds)}")
            
        # 4.4 Клонирование
        if dest_path.suffix != ".parquet":
            dest_path = dest_path.with_suffix(".parquet")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
            
        print(f"Клонирование в {dest_path}...")
        ds.clone(dest_path, optimize_by_reorder=True)

    # 5. Создание метаданных Kaggle
    metadata = {
        "title": "BEBLaDII-Phase1-Resources",
        "id": "bogdanbuliakov/bebladii-resources",
        "licenses": [{"name": "CC0-1.0"}]
    }
    
    with open(staging / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("\nPreparation complete!")
    print(f"All files in: {staging.absolute()}")
    print("Now you can run 'kaggle datasets create -p kaggle_upload'")

if __name__ == "__main__":
    prepare_kaggle_package()
