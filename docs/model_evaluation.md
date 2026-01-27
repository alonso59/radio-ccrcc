# Evaluation Pipeline Usage Guide

## Overview
The evaluation pipeline provides k-fold cross-validation metrics and test set evaluation for trained autoencoder models.

## Structure

```
src/evaluation/
├── __init__.py           # Module exports
├── metrics.py            # PSNR, SSIM, and metrics calculator
├── evaluator.py          # ModelEvaluator for train/val/test evaluation
└── test_loader.py        # TestLoaderFactory for lazy test loading
```

## Key Features

### 1. **Modular Design**
- Separate components for metrics, evaluation, and data loading
- Easy to extend with new metrics or evaluation strategies

### 2. **Memory-Efficient Test Loading**
- Test data is loaded **lazily** only when needed
- Immediate cleanup after evaluation to free memory
- Configurable batch size and workers for test set

### 3. **Comprehensive Metrics**
- **Reconstruction Loss** (MSE)
- **KL Divergence** (β-weighted)
- **Total Loss** (ELBO)
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)

### 4. **Aggregated Statistics**
- Per-fold results
- Cross-fold mean and standard deviation
- Support for train, validation, and test sets

## Usage

### Basic Evaluation (Train + Val)

```python
python evaluation.py
```

### With Test Set Evaluation

Edit `evaluation.py` and set:
```python
evaluate_test_set = True
```

### Configuration

Update these parameters in `evaluation.py`:

```python
outputs_dir = "outputs/2026-01-27/"  # Your experiment date
evaluate_test_set = True              # Enable/disable test evaluation
test_batch_size = 16                  # Batch size for test (lower = less memory)
```

## Output

### Evaluation Summary JSON
Located at: `outputs/<date>/evaluation_summary.json`

Structure:
```json
{
  "folds": [
    {
      "fold": 0,
      "train": {"total_loss": 0.123, "PSNR": 28.5, "SSIM": 0.89, ...},
      "val": {"total_loss": 0.145, "PSNR": 27.2, "SSIM": 0.87, ...},
      "test": {"total_loss": 0.142, "PSNR": 27.5, "SSIM": 0.88, ...}
    },
    ...
  ],
  "mean_train": {"total_loss": 0.125, ...},
  "std_train": {"total_loss": 0.003, ...},
  "mean_val": {"total_loss": 0.148, ...},
  "std_val": {"total_loss": 0.005, ...},
  "mean_test": {"total_loss": 0.145, ...},
  "std_test": {"total_loss": 0.004, ...}
}
```

## Memory Optimization

The pipeline implements several strategies to minimize memory usage:

1. **Sequential Fold Processing**: Only one fold loaded at a time
2. **Lazy Test Loading**: Test data loaded on-demand, then freed
3. **Reduced Workers**: Test loader uses `num_workers=0`
4. **Smaller Batch Size**: Configurable test batch size (default: 16)
5. **Explicit Cleanup**: `del` + `torch.cuda.empty_cache()` after each fold

## Extending the Pipeline

### Add New Metrics

Edit `src/evaluation/metrics.py`:

```python
def my_custom_metric(x_hat, x):
    # Your metric implementation
    return metric_value

class MetricsCalculator:
    def __init__(self):
        self.metrics_functions = {
            "PSNR": psnr_torch,
            "SSIM": ssim_torch,
            "CustomMetric": my_custom_metric  # Add here
        }
```

### Change Loss Function

Edit `evaluation.py`:

```python
from src.trainers.loss import DirichletELBOLoss  # Different loss

# In evaluate_single_fold():
criterion = DirichletELBOLoss()  # Use different criterion
```

### Evaluate Different Model

Edit `evaluation.py`:

```python
from src.models.dirvae import DirVAE  # Different model

# In evaluate_single_fold():
model = DirVAE(cfg).to(device)
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `test_batch_size` (try 8 or 4)
- Set `evaluate_test_set = False` to skip test evaluation
- Ensure test loader uses `num_workers=0`

### Checkpoint Not Found
- Verify experiment folder structure:
  ```
  outputs/<date>/<exp>/
  ├── .hydra/config.yaml
  └── checkpoints/best_model.pth
  ```

### Metric Computation Errors
- Check tensor shapes and data ranges
- Ensure model outputs (reconstruction, z_mu, z_sigma) are correct format
- Verify criterion signature matches model outputs

## Example Workflow

```bash
# 1. Train models with k-fold CV
python main.py dataset.fold=0
python main.py dataset.fold=1
python main.py dataset.fold=2
python main.py dataset.fold=3
python main.py dataset.fold=4

# 2. Evaluate all folds
python evaluation.py

# 3. Check results
cat outputs/2026-01-27/evaluation_summary.json
```
