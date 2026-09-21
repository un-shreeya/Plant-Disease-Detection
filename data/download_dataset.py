import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from PIL import Image
    from tqdm import tqdm
except ImportError:
    print("ERROR: Missing dependencies. Install with:")
    print("  pip install Pillow tqdm")
    sys.exit(1)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MIN_IMAGE_SIZE = 10
RANDOM_SEED = 42

KAGGLE_DATASET = "abdallahalidev/plantvillage-dataset"


def download_from_kaggle(raw_dir: str) -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        import kaggle
    except Exception as e:
        print("\n" + "=" * 70)
        print("  KAGGLE API NOT CONFIGURED")
        print("=" * 70)
        print(f"Error: {e}")
        print("\nTo download automatically, you need to set up Kaggle API:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New Token' under the API section")
        print("  3. Put your KAGGLE_USERNAME and KAGGLE_KEY into the .env file")
        print("\nAlternatively, download manually:")
        print(f"  1. Go to https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
        print("  2. Click 'Download' button")
        print(f"  3. Extract to {raw_dir}/PlantVillage")
        print(f"  4. Run: python data/download_dataset.py --manual --raw_dir {raw_dir}/PlantVillage")
        print("=" * 70)
        sys.exit(1)

    os.makedirs(raw_dir, exist_ok=True)

    print(f"\n📥 Downloading PlantVillage dataset from Kaggle...")
    print(f"   Dataset: {KAGGLE_DATASET}")
    print(f"   Target:  {raw_dir}\n")

    os.system(f'{sys.executable} -m kaggle datasets download -d {KAGGLE_DATASET} -p {raw_dir} --unzip --force')

    possible_paths = [
        os.path.join(raw_dir, "plantvillage dataset", "color"),
        os.path.join(raw_dir, "PlantVillage"),
        os.path.join(raw_dir, "plantvillage"),
        os.path.join(raw_dir, "color"),
        raw_dir,
    ]

    for path in possible_paths:
        if os.path.isdir(path):
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            if len(subdirs) >= 10:
                print(f"   ✅ Found dataset at: {path}")
                return path

    print(f"\n❌ Could not find dataset in {raw_dir}")
    print("   Please check the download and try with --manual flag")
    sys.exit(1)


def validate_image(image_path: str) -> bool:
    try:
        with Image.open(image_path) as img:
            img.verify()

        with Image.open(image_path) as img:
            width, height = img.size
            if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                return False
            return True
    except Exception:
        return False


def create_splits(source_dir: str, output_dir: str):
    print("\n📂 Creating train/val/test splits...")
    print(f"   Source:  {source_dir}")
    print(f"   Output:  {output_dir}")
    print(f"   Split:   {TRAIN_RATIO*100:.0f}% / {VAL_RATIO*100:.0f}% / {TEST_RATIO*100:.0f}%\n")

    random.seed(RANDOM_SEED)

    classes = sorted([
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ])

    if len(classes) == 0:
        print(f"   ❌ No class directories found in {source_dir}")
        sys.exit(1)

    print(f"   Found {len(classes)} classes\n")

    total_stats = {"train": 0, "val": 0, "test": 0, "skipped": 0}
    class_mapping = {}

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    for idx, class_name in enumerate(tqdm(classes, desc="   Processing classes")):
        class_mapping[idx] = class_name
        class_dir = os.path.join(source_dir, class_name)

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        all_files = [
            f for f in os.listdir(class_dir)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]

        valid_files = []
        for f in all_files:
            filepath = os.path.join(class_dir, f)
            if validate_image(filepath):
                valid_files.append(f)
            else:
                total_stats["skipped"] += 1

        random.shuffle(valid_files)

        n = len(valid_files)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)

        splits = {
            "train": valid_files[:train_end],
            "val": valid_files[train_end:val_end],
            "test": valid_files[val_end:],
        }

        for split_name, files in splits.items():
            split_class_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            for f in files:
                src = os.path.join(class_dir, f)
                dst = os.path.join(split_class_dir, f)
                shutil.copy2(src, dst)

            total_stats[split_name] += len(files)

    mapping_path = os.path.join(os.path.dirname(output_dir), "class_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(class_mapping, f, indent=2)

    print("\n" + "=" * 60)
    print("  📊 DATASET SPLIT SUMMARY")
    print("=" * 60)
    print(f"  Classes:     {len(classes)}")
    print(f"  Train:       {total_stats['train']:,} images")
    print(f"  Validation:  {total_stats['val']:,} images")
    print(f"  Test:        {total_stats['test']:,} images")
    print(f"  Skipped:     {total_stats['skipped']:,} (corrupt/invalid)")
    total = total_stats['train'] + total_stats['val'] + total_stats['test']
    print(f"  Total Valid: {total:,} images")
    print(f"\n  Class mapping saved to: {mapping_path}")
    print("=" * 60)

    return class_mapping


def print_class_distribution(output_dir: str):
    print("\n📊 Class Distribution:\n")
    print(f"  {'Class Name':<45} {'Train':>7} {'Val':>7} {'Test':>7}")
    print("  " + "-" * 70)

    for split in ["train"]:
        split_dir = os.path.join(output_dir, split)
        if not os.path.exists(split_dir):
            continue

        classes = sorted(os.listdir(split_dir))
        for class_name in classes:
            train_count = len(os.listdir(os.path.join(output_dir, "train", class_name))) \
                if os.path.exists(os.path.join(output_dir, "train", class_name)) else 0
            val_count = len(os.listdir(os.path.join(output_dir, "val", class_name))) \
                if os.path.exists(os.path.join(output_dir, "val", class_name)) else 0
            test_count = len(os.listdir(os.path.join(output_dir, "test", class_name))) \
                if os.path.exists(os.path.join(output_dir, "test", class_name)) else 0

            display_name = class_name[:43] + ".." if len(class_name) > 45 else class_name
            print(f"  {display_name:<45} {train_count:>7} {val_count:>7} {test_count:>7}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare PlantVillage dataset for training"
    )
    parser.add_argument(
        "--manual", action="store_true",
        help="Skip download, use already-downloaded data in --raw_dir"
    )
    parser.add_argument(
        "--raw_dir", type=str, default="data/raw/PlantVillage",
        help="Path to raw PlantVillage images (for --manual mode)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/processed",
        help="Output directory for train/val/test splits"
    )
    args = parser.parse_args()

    print("\n🌿 Plant Disease Detection — Dataset Preparation")
    print("=" * 50)

    if args.manual:
        if not os.path.isdir(args.raw_dir):
            print(f"\n❌ Directory not found: {args.raw_dir}")
            print("\nPlease download the dataset first:")
            print(f"  1. Go to https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
            print(f"  2. Download and extract to {args.raw_dir}")
            sys.exit(1)
        source_dir = args.raw_dir
    else:
        source_dir = download_from_kaggle("data/raw")

    class_mapping = create_splits(source_dir, args.output_dir)

    print_class_distribution(args.output_dir)

    print("\n✅ Dataset preparation complete!")
    print("   Next step: python models/train.py")


"""
=============================================================================
 Dataset Download & Preparation Script
 File: data/download_dataset.py
=============================================================================
 PURPOSE:
   Downloads the PlantVillage dataset from Kaggle, validates images,
   and splits into train/validation/test sets (70/15/15).

 USAGE:
   # Option 1: With Kaggle API credentials in .env
   python data/download_dataset.py

   # Option 2: Manual download
   python data/download_dataset.py --manual --raw_dir data/raw/PlantVillage

 WHAT IT DOES:
   1. Downloads PlantVillage dataset (54,303 images, 38 classes)
   2. Validates all images (removes corrupt/unreadable files)
   3. Splits into train (70%), validation (15%), test (15%)
   4. Creates class_mapping.json with class index → disease name
   5. Prints dataset statistics and class distribution

 OUTPUT STRUCTURE:
   data/
   ├── raw/PlantVillage/          # Original downloaded images
   ├── processed/
   │   ├── train/                  # 70% of images
   │   │   ├── Apple___Apple_scab/
   │   │   ├── Apple___Black_rot/
   │   │   └── ...
   │   ├── val/                    # 15% of images
   │   │   └── ...
   │   └── test/                   # 15% of images
   │       └── ...
   └── class_mapping.json          
=============================================================================
"""
    print()
