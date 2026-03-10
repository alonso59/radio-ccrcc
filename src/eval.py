"""
K-Fold Cross-Validation Evaluation Pipeline
Evaluates trained models on train, validation, and test sets.
"""

import os
import json
import torch
import logging
from omegaconf import OmegaConf
from pathlib import Path
from typing import Optional

# Import evaluation components
from evaluation import ModelEvaluator
from evaluation.test_loader import TestLoaderFactory

# Import training components
from torch.nn import L1Loss
from monai.losses.perceptual import PerceptualLoss
from dataloader.dataloader import DataLoaderFactory
from models.autoencoder import Autoencoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_fold_experiments(results_base_dir: str) -> dict:
	"""
	Scan results directory for fold checkpoints via 'latest' symlinks.
	
	Args:
		results_base_dir: Base results directory (e.g., results/Dataset320_ldm_vae)
		
	Returns:
		Dictionary mapping fold_id -> experiment_path
	"""
	fold_experiments = {}
	
	# Iterate over fold_N directories
	base_path = Path(results_base_dir)
	for fold_dir in sorted(base_path.glob("fold_*")):
		if not fold_dir.is_dir():
			continue
		
		try:
			# Extract fold number from directory name
			fold_id = int(fold_dir.name.split("_")[1])
			
			# Look for 'latest' symlink first (recommended approach)
			latest_link = fold_dir / "latest"
			
			if latest_link.exists():
				run_dir = latest_link.resolve()
				logger.info(f"Found fold {fold_id} via 'latest' symlink: {run_dir.name}")
			else:
				# Fallback: find most recent timestamped run
				logger.warning(f"No 'latest' symlink for {fold_dir.name}, scanning timestamps...")
				run_dir = _find_most_recent_run(fold_dir)
				if run_dir is None:
					logger.warning(f"No valid runs found in {fold_dir.name}")
					continue
			
			# Validate required files exist
			checkpoint_path = run_dir / "checkpoint_best.pth"
			config_path = run_dir / ".hydra" / "config.yaml"
			
			if checkpoint_path.exists() and config_path.exists():
				fold_experiments[fold_id] = {
					'path': str(run_dir),
					'config_path': str(config_path),
					'checkpoint_path': str(checkpoint_path)
				}
				logger.info(f"✓ Fold {fold_id}: {run_dir.relative_to(base_path)}")
			else:
				missing = []
				if not checkpoint_path.exists():
					missing.append("checkpoint_best.pth")
				if not config_path.exists():
					missing.append(".hydra/config.yaml")
				logger.warning(f"✗ Fold {fold_id}: Missing {', '.join(missing)} in {run_dir}")
			
		except Exception as e:
			logger.error(f"Error processing {fold_dir.name}: {e}")
			continue
	
	return fold_experiments


def _find_most_recent_run(fold_dir: Path) -> Optional[Path]:
	"""
	Fallback: Find most recent timestamped run directory with valid checkpoint.
	
	Args:
		fold_dir: Path to fold_N directory
		
	Returns:
		Path to most recent run directory, or None if no valid runs found
	"""
	runs = []
	
	# Scan for YYYY-MM-DD/HH-MM-SS structure
	for date_dir in fold_dir.glob("*/"):
		if date_dir.name == "latest":
			continue
		for time_dir in date_dir.glob("*/"):
			checkpoint = time_dir / "checkpoint_best.pth"
			if checkpoint.exists():
				runs.append((checkpoint.stat().st_mtime, time_dir))
	
	if not runs:
		return None
	
	# Sort by modification time (most recent first)
	runs.sort(reverse=True, key=lambda x: x[0])
	return runs[0][1]


def evaluate_single_fold(
	fold_id: int,
	experiment_info: dict,
	device: torch.device,
	evaluate_test: bool = False,
	test_batch_size: int = 16
) -> dict:
	"""
	Evaluate a single fold with train, val, and optionally test sets.
	
	Args:
		fold_id: Fold identifier
		experiment_info: Dictionary with paths to config and checkpoint
		device: Torch device
		evaluate_test: Whether to evaluate on test set (lazy loading)
		test_batch_size: Batch size for test evaluation
		
	Returns:
		Fold evaluation results dictionary
	"""
	logger.info(f"\n{'='*80}")
	logger.info(f"EVALUATING FOLD {fold_id}")
	logger.info(f"{'='*80}")
	
	# Load config
	cfg = OmegaConf.load(experiment_info['config_path'])
	
	# Load model
	model = Autoencoder(cfg).to(device)
	checkpoint = torch.load(experiment_info['checkpoint_path'], map_location=device)
	
	# Handle different checkpoint formats
	if 'model_state_dict' in checkpoint:
		model.load_state_dict(checkpoint['model_state_dict'])
	else:
		model.load_state_dict(checkpoint)
	
	logger.info(f"Loaded checkpoint from: {experiment_info['checkpoint_path']}")
	
	# Create train and val loaders
	train_loader, val_loader, normalization_stats = DataLoaderFactory.create_loaders(cfg.dataset)
	
	# Initialize criteria and evaluator (matching trainer.py)
	l1_loss = L1Loss()
	perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
	perceptual_loss.to(device)
	evaluator = ModelEvaluator(
		device=device,
		l1_loss=l1_loss,
		perceptual_loss=perceptual_loss,
		normalization_stats=normalization_stats
	)
	
	# Prepare test loader if requested
	test_loader = None
	if evaluate_test:
		logger.info("Creating test loader (lazy loading)...")
		from dataloader.augmentations import val_augmentations
		from hydra.utils import to_absolute_path

		# Construct splits_path same way as DataLoaderFactory
		dataset_id = cfg.dataset.dataset_id
		base_path = cfg.dataset.get('base_path', 'data/dataset')
		base_path = base_path if os.path.isabs(base_path) else to_absolute_path(base_path)
		splits_path = os.path.join(base_path, dataset_id, 'voi', 'splits_final.json')
		
		# Convert normalization_stats dict to tuple for val_augmentations
		data_stats = (
			normalization_stats['mean'],
			normalization_stats['std'],
			normalization_stats['median'],
			normalization_stats['p25'],
			normalization_stats['p75']
		)
		val_transform = val_augmentations(data_stats)
		
		test_loader, num_test = TestLoaderFactory.create_test_loader(
			splits_path=splits_path,
			val_transform=val_transform,
			batch_size=test_batch_size,
			num_workers=0  # Low workers to save memory
		)
		logger.info(f"Test loader created with {num_test} samples")
	
	# Evaluate fold
	fold_result = evaluator.evaluate_fold(
		model=model,
		train_loader=train_loader,
		val_loader=val_loader,
		test_loader=test_loader,
		fold_id=fold_id,
		kl_weight=cfg.loss_weights.kl,
		perceptual_weight=cfg.loss_weights.perceptual
	)
	
	# Clean up
	del model, train_loader, val_loader
	torch.cuda.empty_cache() if torch.cuda.is_available() else None
	
	return fold_result


def main():
	"""
	Main evaluation pipeline for k-fold cross-validation.
	
	Scans a results experiment directory for trained fold checkpoints and evaluates them.
	Uses 'latest' symlinks to discover the most recent run for each fold.
	
	Usage:
		python -m src.evaluation
		
		Or edit results_dir below to point to your experiment:
		  results/Dataset320_ldm_vae
		  results/Dataset420_ldm_gan
	"""
	# Configuration
	results_dir = "results/Dataset620_ldm_vae"  # Update to your experiment path
	evaluate_test_set = True  # Set to True to evaluate test set per fold
	test_batch_size = 1  # Must be 1 for variable-sized test volumes
	
	# Device setup
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logger.info(f"Using device: {device}")
	
	# Find all fold experiments
	logger.info(f"Scanning directory: {results_dir}")
	fold_experiments = find_fold_experiments(results_dir)
	
	if not fold_experiments:
		logger.error("No valid fold experiments found!")
		return
	
	logger.info(f"Found {len(fold_experiments)} fold experiments")
	
	# Evaluate each fold
	fold_results = []
	for fold_id in sorted(fold_experiments.keys()):
		try:
			fold_result = evaluate_single_fold(
				fold_id=fold_id,
				experiment_info=fold_experiments[fold_id],
				device=device,
				evaluate_test=evaluate_test_set,
				test_batch_size=test_batch_size
			)
			fold_results.append(fold_result)
		except Exception as e:
			logger.error(f"Error evaluating fold {fold_id}: {e}")
			continue
	
	if not fold_results:
		logger.error("No successful fold evaluations!")
		return
	
	# Aggregate results across folds
	logger.info("\n" + "="*80)
	logger.info("AGGREGATING RESULTS")
	logger.info("="*80)
	
	# Create a simple evaluator instance for aggregation (no need for losses)
	from torch.nn import L1Loss
	from monai.losses.perceptual import PerceptualLoss
	
	dummy_l1 = L1Loss()
	dummy_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
	aggregator = ModelEvaluator(
		device=device,
		l1_loss=dummy_l1,
		perceptual_loss=dummy_perceptual,
		normalization_stats={}
	)
	summary = aggregator.aggregate_folds(fold_results)
	
	# Print summary
	logger.info("\n" + "="*80)
	logger.info("CROSS-VALIDATION SUMMARY")
	logger.info("="*80)
	
	for split in ['train', 'val', 'test']:
		mean_key = f'mean_{split}'
		std_key = f'std_{split}'
		
		if mean_key in summary:
			logger.info(f"\n{split.upper()} SET:")
			for metric, value in summary[mean_key].items():
				std_value = summary[std_key][metric]
				logger.info(f"  {metric}: {value:.4f} ± {std_value:.4f}")
	
	# Save summary to JSON
	output_path = os.path.join(results_dir, "evaluation_summary.json")
	with open(output_path, "w") as f:
		json.dump(summary, f, indent=2)
	
	logger.info(f"\nSaved evaluation summary to: {output_path}")
	logger.info("Evaluation complete!")

if __name__ == "__main__":
	main()
