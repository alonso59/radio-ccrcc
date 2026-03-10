"""
Training module for radio-ccrcc project.
Provides unified training infrastructure following SOLID principles.
"""

# Base trainer
from .base_trainer import BaseTrainer

# Concrete trainers
from .trainer import AutoencoderTrainer
from .classifier_trainer import ClassifierTrainer

# Factory
from .trainer_factory import TrainerFactory

# Legacy trainers (deprecated)
# from .trainer import Trainer
# from .trainer_gan import TrainerGAN
# from .trainer_classifier import TrainerClassifierVAESpatial

__all__ = [
    'BaseTrainer',
    'AutoencoderTrainer',
    'ClassifierTrainer',
    'TrainerFactory',
]
