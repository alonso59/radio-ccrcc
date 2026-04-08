"""
Training Orchestrator - Coordinates the entire training process.
Follows Facade Pattern and Single Responsibility Principle.
"""
import logging

from omegaconf import DictConfig, OmegaConf

from ..utils.stdout_logger import setup_stdout_logging
from .config_manager import ConfigManager
from .setup_builder import SetupBuilderDirector

logger = logging.getLogger(__name__)

class TrainingOrchestrator:
    """
    Orchestrates the complete training workflow.
    Provides a simplified interface to the training subsystem.
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.config_manager = ConfigManager(cfg)
        self.setup = None
    
    def initialize(self) -> None:
        """Initialize the training environment."""
        # Setup logging
        setup_stdout_logging(filter_stderr_tqdm=True)
        
        # Display configuration
        self._display_configuration()
        
        # Validate and prepare configuration
        self.config_manager.validate()
        self.config_manager.resolve_paths()
    
    def _display_configuration(self) -> None:
        """Display training configuration."""
        logging.info("=" * 80)
        logging.info("Training Configuration")
        logging.info("=" * 80)
        logging.info(OmegaConf.to_yaml(self.cfg))
        logging.info("=" * 80)
    
    def prepare_training(self) -> None:
        """Prepare all training components."""
        training_mode = self.config_manager.get_training_mode()
        logging.info(f"\n🚀 Starting training in '{training_mode}' mode...\n")
        
        # Build training setup using director
        director = SetupBuilderDirector(self.cfg, self.config_manager)
        self.setup = director.construct_setup(training_mode)
    
    def execute_training(self) -> None:
        """Execute the training process."""
        if self.setup is None:
            raise RuntimeError("Training not prepared. Call prepare_training() first.")
        
        try:
            self.setup.trainer.fit()
            logging.info("\n✅ Training completed successfully!")
        except KeyboardInterrupt:
            logging.info("\n⚠️  Training interrupted by user")
        except Exception as e:
            logging.info(f"\n❌ Training failed with error: {e}")
            raise
        finally:
            self._cleanup()
    
    def _cleanup(self) -> None:
        """Cleanup resources."""
        if self.setup and self.setup.logger:
            self.setup.logger.close()
            logger.debug("Logger closed.")
    
    def run(self) -> None:
        """Run complete training workflow."""
        self.initialize()
        self.prepare_training()
        self.execute_training()
