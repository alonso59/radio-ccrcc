"""
Main Training Script - Entry point for model training.

This module provides a clean entry point for training various models (VAE, GAN, Classifier).
It uses the Orchestrator pattern to coordinate the training workflow and leverages
Hydra for configuration management.

Architecture:
    - ConfigManager: Handles configuration validation and preprocessing
    - ComponentFactory: Creates models, loggers, callbacks, and dataloaders
    - SetupBuilder: Builds training setups using Builder pattern
    - TrainingOrchestrator: Coordinates the entire training workflow

Usage:
    python main.py trainer.training_mode=vae
    python main.py trainer.training_mode=gan
    python main.py trainer.training_mode=classifier
"""
import hydra
from omegaconf import DictConfig

from src.setup.orchestrator import TrainingOrchestrator


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training.
    
    Creates and runs a TrainingOrchestrator which handles:
    - Configuration validation and preprocessing
    - Component creation (models, dataloaders, loggers, callbacks)
    - Training execution and error handling
    - Resource cleanup
    
    Args:
        cfg: Hydra configuration object loaded from config/config.yaml
    """
    orchestrator = TrainingOrchestrator(cfg)
    orchestrator.run()


if __name__ == "__main__":
    main()
