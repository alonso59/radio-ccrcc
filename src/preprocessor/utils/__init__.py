"""Utility modules: I/O operations, metrics computation, and common helpers."""

from .common import (
    TUMOR_LABEL,
    BACKGROUND_HU,
    SUPPORTED_INPUT_FORMATS,
    extract_patient_id,
    categorize_files_by_type,
    get_files_by_type_for_patients,
)
from .io import (
    detect_input_format,
    discover_files,
    load_case,
    save_voi,
)
from .metrics import (
    compute_tumor_metrics,
    collect_fingerprint_data,
    generate_fingerprint,
    compute_fingerprint_from_outputs,
)

__all__ = [
    # Common utilities
    'TUMOR_LABEL',
    'BACKGROUND_HU',
    'SUPPORTED_INPUT_FORMATS',
    'extract_patient_id',
    'categorize_files_by_type',
    'get_files_by_type_for_patients',
    # I/O
    'detect_input_format',
    'discover_files',
    'load_case',
    'save_voi',
    # Metrics
    'compute_tumor_metrics',
    'collect_fingerprint_data',
    'generate_fingerprint',
    'compute_fingerprint_from_outputs',
]
