#!/usr/bin/env python3
"""
Generate train/val/test splits from folder structure.

Usage:
    python generate_splits.py <data_folder> [options]
    
Expected structure:
    data_folder/images/{class}/{case_folder}/*.npy
    data_folder/segmentation/{class}/{case_folder}/*.npy  (optional)
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def extract_patient_id(filepath: Path) -> str:
    """
    Extract patient ID from filepath.
    
    Examples:
        .../AB/case_00093/01_case_00093_L.npy -> case_00093
        .../AB/TCGA-XX-YYYY/TCGA-XX-YYYY_slice.npy -> TCGA-XX-YYYY
    """
    case_folder = filepath.parent.name
    
    # Handle case_XXXXX format
    if case_folder.startswith('case_'):
        return case_folder
    
    # Handle TCGA format
    if case_folder.startswith('TCGA-'):
        return case_folder
    
    # Fallback: extract from filename
    stem = filepath.stem
    if 'case_' in stem:
        parts = stem.split('_')
        for i, part in enumerate(parts):
            if part == 'case' and i + 1 < len(parts):
                return f"case_{parts[i+1]}"
    
    return case_folder


def scan_dataset(data_folder: Path, subfolder: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Scan dataset folder and organize files by patient and class.
    
    Args:
        data_folder: Root data folder
        subfolder: 'images' or 'segmentation'
        
    Returns:
        patient_files: {patient_id: [file_paths]}
        patient_classes: {patient_id: class_label}
    """
    target_folder = data_folder / subfolder
    
    if not target_folder.exists():
        print(f"Error: {target_folder} does not exist")
        sys.exit(1)
    
    patient_files = defaultdict(list)
    patient_classes = {}
    class_counts = defaultdict(int)
    
    # Scan class folders
    class_folders = sorted([f for f in target_folder.iterdir() if f.is_dir()])
    
    if not class_folders:
        print(f"Error: No class folders found in {target_folder}")
        sys.exit(1)
    
    print(f"\nScanning {subfolder}/:")
    
    for class_folder in class_folders:
        class_label = class_folder.name
        files = list(class_folder.rglob('*.npy')) + list(class_folder.rglob('*.nii.gz'))
        
        for filepath in files:
            patient_id = extract_patient_id(filepath)
            abs_path = str(filepath.resolve())
            
            patient_files[patient_id].append(abs_path)
            patient_classes[patient_id] = class_label
        
        class_counts[class_label] += len(set(extract_patient_id(f) for f in files))
    
    print(f"  Found {len(patient_files)} patients")
    print(f"  Class distribution: {dict(class_counts)}")
    
    return dict(patient_files), patient_classes


def create_splits(
    patient_files: Dict[str, List[str]],
    patient_classes: Dict[str, str],
    test_ratio: float,
    n_folds: int,
    seed: int
) -> Dict:
    """
    Create train/val/test splits with stratification.
    
    Args:
        patient_files: Patient ID to file paths mapping
        patient_classes: Patient ID to class label mapping
        test_ratio: Fraction for test set
        n_folds: Number of cross-validation folds
        seed: Random seed
        
    Returns:
        Splits dictionary
    """
    patients = list(patient_files.keys())
    classes = [patient_classes[p] for p in patients]
    
    # Train/test split (stratified by class)
    try:
        train_patients, test_patients = train_test_split(
            patients,
            test_size=test_ratio,
            stratify=classes,
            random_state=seed
        )
        print(f"\n✓ Stratified train/test split: {len(train_patients)} train, {len(test_patients)} test")
    except ValueError as e:
        print(f"\n⚠ Stratification failed ({e}), using random split")
        rng = np.random.default_rng(seed)
        shuffled = patients.copy()
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_ratio))
        test_patients = shuffled[:n_test]
        train_patients = shuffled[n_test:]
    
    # K-fold cross-validation on training set
    train_classes = [patient_classes[p] for p in train_patients]
    folds = []
    
    try:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        train_arr = np.array(train_patients)
        class_arr = np.array(train_classes)
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_arr, class_arr)):
            fold_train_patients = train_arr[train_idx].tolist()
            fold_val_patients = train_arr[val_idx].tolist()
            
            fold_train_files = [f for p in fold_train_patients for f in patient_files[p]]
            fold_val_files = [f for p in fold_val_patients for f in patient_files[p]]
            
            folds.append({
                'fold': fold_idx,
                'train_case_ids': fold_train_patients,
                'val_case_ids': fold_val_patients,
                'train_ct_files': fold_train_files,
                'val_ct_files': fold_val_files,
            })
        
        print(f"✓ Created {n_folds} stratified folds")
    except ValueError:
        print(f"⚠ Stratified K-fold failed, using regular K-fold")
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        train_arr = np.array(train_patients)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_arr)):
            fold_train_patients = train_arr[train_idx].tolist()
            fold_val_patients = train_arr[val_idx].tolist()
            
            fold_train_files = [f for p in fold_train_patients for f in patient_files[p]]
            fold_val_files = [f for p in fold_val_patients for f in patient_files[p]]
            
            folds.append({
                'fold': fold_idx,
                'train_case_ids': fold_train_patients,
                'val_case_ids': fold_val_patients,
                'train_ct_files': fold_train_files,
                'val_ct_files': fold_val_files,
            })
    
    # Test set files
    test_files = [f for p in test_patients for f in patient_files[p]]
    
    return {
        'params': {
            'n_folds': n_folds,
            'test_ratio': test_ratio,
            'random_seed': seed,
        },
        'ct_folds': folds,
        'test_case_ids': test_patients,
        'test_ct_files': test_files,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate train/val/test splits from folder structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config file (recommended)
  python generate_splits.py --config config/planner.yaml
  
  # Direct path (legacy)
  python generate_splits.py data/dataset/tcga_kirc_tumor
  python generate_splits.py data/dataset/tcga_kirc_tumor --test-ratio 0.15 --n-folds 3
        """
    )
    parser.add_argument('data_folder', type=Path, nargs='?', help='Data folder (containing images/ or segmentation/)')
    parser.add_argument('--config', type=Path, help='Config file (overrides data_folder if provided)')
    parser.add_argument('--test-ratio', type=float, help='Test set ratio (default: from config or 0.2)')
    parser.add_argument('--n-folds', type=int, help='Number of CV folds (default: from config or 5)')
    parser.add_argument('--seed', type=int, help='Random seed (default: from config or 42)')
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Derive data_folder from config
        if config.get('OUTPUT_DIR'):
            data_folder = Path(config['OUTPUT_DIR'])
        elif config.get('DATASET_ID'):
            output_base = config.get('OUTPUT_BASE', 'data/dataset')
            dataset_id = config['DATASET_ID']
            data_folder = Path(output_base) / f"Dataset{dataset_id}" / "voi"
        else:
            print("Error: Config must have OUTPUT_DIR or DATASET_ID")
            sys.exit(1)
        
        # Get split parameters from config (with fallbacks)
        test_ratio = args.test_ratio if args.test_ratio is not None else config.get('TEST_RATIO', 0.2)
        n_folds = args.n_folds if args.n_folds is not None else config.get('N_FOLDS', 5)
        seed = args.seed if args.seed is not None else config.get('RANDOM_SEED', 42)
    else:
        # Legacy mode: data_folder required
        if not args.data_folder:
            parser.error("Either --config or data_folder must be provided")
        data_folder = args.data_folder
        test_ratio = args.test_ratio if args.test_ratio is not None else 0.2
        n_folds = args.n_folds if args.n_folds is not None else 5
        seed = args.seed if args.seed is not None else 42
    
    if not data_folder.exists():
        print(f"Error: {data_folder} does not exist")
        sys.exit(1)
    
    print(f"{'='*60}")
    print(f"Generate Splits")
    print(f"{'='*60}")
    print(f"Data folder: {data_folder}")
    print(f"Test ratio: {test_ratio}")
    print(f"K-folds: {n_folds}")
    print(f"Seed: {seed}")
    
    # Scan images folder (primary)
    patient_files, patient_classes = scan_dataset(data_folder, 'images')
    
    # Create splits
    splits = create_splits(
        patient_files,
        patient_classes,
        test_ratio,
        n_folds,
        seed
    )
    
    # Save splits_images.json
    output_path = data_folder / 'splits_images.json'
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"\n✓ Saved: {output_path}")
    
    # If segmentation folder exists, create splits_segmentation.json
    if (data_folder / 'segmentation').exists():
        seg_patient_files, _ = scan_dataset(data_folder, 'segmentation')
        seg_splits = create_splits(
            seg_patient_files,
            patient_classes,
            test_ratio,
            n_folds,
            seed
        )
        
        seg_output_path = data_folder / 'splits_segmentation.json'
        with open(seg_output_path, 'w') as f:
            json.dump(seg_splits, f, indent=2)
        print(f"✓ Saved: {seg_output_path}")
    
    print(f"\n{'='*60}")
    print(f"✓ Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
