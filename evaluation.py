import os
import json
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from training.loss import AutoencoderLoss
from pytorch_msssim import ssim
from training.dataloader import DataLoaderFactory

# Import model class (assume autoencoder for now, can be extended)
from models.autoencoder import Autoencoder

def psnr_torch(x_hat, x, data_range=1.0):
	mse = torch.mean((x_hat - x) ** 2)
	if mse == 0:
		return torch.tensor(float('inf'), device=x_hat.device)
	psnr = 20 * torch.log10(torch.tensor(data_range, device=x_hat.device)) - 10 * torch.log10(mse)
	return psnr

def evaluate_model(model, dataloader, device, criterion, metrics):
	model.eval()
	results = {k: [] for k in ['loss'] + list(metrics.keys())}
	with torch.no_grad():
		for batch in tqdm(dataloader, desc="Evaluating", ncols=100, ascii=True):
			inputs = batch['ct']['data'].to(device)
			reconstruction, z_mu, z_sigma = model(inputs)
			loss = criterion(reconstruction, inputs, mu=z_mu, logvar=z_sigma, beta=1.0)
			results['loss'].append(loss.item())
			for m_name, m_fn in metrics.items():
				metric_value = m_fn(reconstruction, inputs)
				if hasattr(metric_value, 'item'):
					metric_value = metric_value.item()
				results[m_name].append(metric_value)
	# Compute mean for each metric
	return {k: float(np.mean(v)) for k, v in results.items()}

def main():
	outputs_dir = "outputs/2025-09-02/"
	summary = {}
	fold_metrics = []
	# Find all experiment folders with checkpoints
	for date_folder in sorted(os.listdir(outputs_dir)):
		date_path = os.path.join(outputs_dir)
		if not os.path.isdir(date_path):
			continue
		for exp_folder in sorted(os.listdir(date_path)):
			exp_path = os.path.join(date_path, exp_folder)
			hydra_cfg_path = os.path.join(exp_path, ".hydra", "config.yaml")
			checkpoint_path = os.path.join(exp_path, "checkpoints", "best_model.pth")
			if not (os.path.exists(hydra_cfg_path) and os.path.exists(checkpoint_path)):
				continue
			# Load config
			cfg = OmegaConf.load(hydra_cfg_path)
			# Load model
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			model = Autoencoder(cfg).to(device)
			# model.load_state_dict(torch.load(checkpoint_path, map_location=device))
			# Build dataloaders
			train_loader, val_loader = DataLoaderFactory.create_loaders(cfg.dataset)
			criterion = AutoencoderLoss()
			metrics = {
				"PSNR": psnr_torch,
				"SSIM": lambda x_hat, x: ssim(x_hat, x, data_range=1.0)
			}
			# Evaluate train and val
			train_metrics = evaluate_model(model, train_loader, device, criterion, metrics)
			val_metrics = evaluate_model(model, val_loader, device, criterion, metrics)
			fold_result = {
				"fold": cfg.dataset.fold,
				"train": train_metrics,
				"val": val_metrics
			}
			fold_metrics.append(fold_result)
			print(f"Fold {cfg.dataset.fold}: train={train_metrics}, val={val_metrics}")
	if not fold_metrics:
		print("No valid folds found.")
		return
	# Aggregate metrics
	summary["folds"] = fold_metrics
	# Compute mean and std for each metric
	all_train = {k: [f['train'][k] for f in fold_metrics] for k in fold_metrics[0]['train']}
	all_val = {k: [f['val'][k] for f in fold_metrics] for k in fold_metrics[0]['val']}
	summary["mean_train"] = {k: float(np.mean(v)) for k, v in all_train.items()}
	summary["std_train"] = {k: float(np.std(v)) for k, v in all_train.items()}
	summary["mean_val"] = {k: float(np.mean(v)) for k, v in all_val.items()}
	summary["std_val"] = {k: float(np.std(v)) for k, v in all_val.items()}
	# Save summary
	with open(os.path.join(outputs_dir, "kfold_summary.json"), "w") as f:
		json.dump(summary, f, indent=2)
	print("Saved kfold_summary.json")

if __name__ == "__main__":
	main()
