"""Shared training-mode normalization helpers."""

from __future__ import annotations

from typing import Dict, List

MODE_ALIASES: Dict[str, str] = {
    "autoencoder": "autoencoder",
    "vae": "autoencoder",
    "gan": "autoencoder",
    "classifier": "classifier",
}


def normalize_training_mode(mode: str) -> str:
    """Map supported aliases to their canonical training mode."""
    normalized = str(mode).lower()
    if normalized not in MODE_ALIASES:
        available = ", ".join(sorted(MODE_ALIASES))
        raise ValueError(f"Unknown training mode: {mode!r}. Available modes: {available}")
    return MODE_ALIASES[normalized]


def supported_training_modes() -> List[str]:
    """Return all accepted training-mode strings."""
    return sorted(MODE_ALIASES)


def legacy_checkpoint_modes(mode: str) -> List[str]:
    """Return legacy checkpoint suffixes to search for a mode."""
    raw_mode = str(mode).lower()
    canonical_mode = normalize_training_mode(raw_mode)
    if canonical_mode == "autoencoder":
        ordered = [raw_mode, "autoencoder", "vae", "gan"]
    else:
        ordered = [raw_mode, canonical_mode]

    unique: List[str] = []
    for item in ordered:
        if item not in unique:
            unique.append(item)
    return unique
