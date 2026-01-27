"""
VOI Preprocessor Package

A modular preprocessing pipeline for extracting kidney VOIs from CT scans.
Supports both .npy and .nii.gz input formats, outputs to .npy.

Output structure:
    {OUTPUT_DIR}/
    ├── mask/{class}/{patient_id}/{case_name}_{side}.npy
    ├── segmentation/{class}/{patient_id}/{case_name}_{side}.npy
    ├── dataset.json
    ├── dataset_fingerprint.json
    └── sanity_check.json

Usage:
    from preprocessor import VOIPreprocessor, load_config
    
    config = load_config('preprocessor_config.yaml')
    pipeline = VOIPreprocessor(config)
    results = pipeline.run_batch()
"""

from .core.config import load_config
from .core.pipeline import VOIPreprocessor
from .analysis.sanity_check import SanityChecker
from .analysis.edge_case_analyzer import EdgeCaseAnalyzer

__all__ = ['VOIPreprocessor', 'load_config', 'SanityChecker', 'EdgeCaseAnalyzer']
__version__ = '1.0.0'
