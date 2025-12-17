"""
Utilities for redirecting stdout/stderr to logging while maintaining console output.
Handles tqdm progress bars intelligently to avoid cluttering log files.
"""
import sys
import logging
from typing import Optional, IO, Any


class StreamToLogger:
    """
    Redirects stream writes to a logger while maintaining original stream output.
    Intelligently filters tqdm progress bars to keep log files clean.
    """
    
    def __init__(
        self, 
        logger: logging.Logger, 
        log_level: int = logging.INFO, 
        stream: Optional[Any] = None, 
        filter_tqdm: bool = False
    ):
        """Initialize stream-to-logger redirect.
        
        Args:
            logger: Logger instance to write to
            log_level: Logging level for messages
            stream: Original stream to maintain (e.g., sys.stdout)
            filter_tqdm: If True, filters out tqdm intermediate progress updates
        """
        self.logger = logger
        self.log_level = log_level
        self.stream = stream
        self.filter_tqdm = filter_tqdm
        self.last_line = ''

    def write(self, buf: str) -> None:
        """Write buffer to both original stream and logger."""
        # Always write to original stream (console) for live monitoring
        if self.stream is not None:
            self.stream.write(buf)
            self.stream.flush()
        
        # Apply tqdm filtering for log file if enabled
        if self.filter_tqdm:
            stripped = buf.strip()
            
            # Skip empty lines and whitespace
            if not stripped:
                return
            
            # Detect tqdm progress bars (contain progress indicators)
            if '|' in stripped and '%' in stripped:
                # Only log completed progress bars
                if '100%' in stripped and stripped != self.last_line:
                    self.logger.log(self.log_level, stripped)
                    self.last_line = stripped
                # Skip intermediate progress updates
                return
        
        # Log all non-empty lines
        for line in buf.rstrip().splitlines():
            line_stripped = line.rstrip()
            if line_stripped:
                self.logger.log(self.log_level, line_stripped)

    def flush(self) -> None:
        """Flush the original stream."""
        if self.stream is not None:
            self.stream.flush()


def setup_stdout_logging(filter_stderr_tqdm: bool = True) -> None:
    """
    Setup stdout/stderr redirection to capture all output in Hydra's main.log.
    
    This ensures that both Python logging and print() statements are captured
    in the log file while still displaying in the terminal.
    
    Args:
        filter_stderr_tqdm: If True, filters tqdm progress bars from stderr logs
                           to keep log files clean (still shows in terminal)
    """
    # Get or create loggers for stdout and stderr
    stdout_logger = logging.getLogger('STDOUT')
    stderr_logger = logging.getLogger('STDERR')
    
    # Redirect stdout - capture everything
    sys.stdout = StreamToLogger(
        stdout_logger, 
        logging.INFO, 
        sys.stdout, 
        filter_tqdm=False
    )
    
    # Redirect stderr - optionally filter tqdm progress bars
    # Use INFO level instead of ERROR for stderr (tqdm outputs there by default)
    sys.stderr = StreamToLogger(
        stderr_logger, 
        logging.INFO,  # Changed from ERROR to INFO
        sys.stderr, 
        filter_tqdm=filter_stderr_tqdm
    )
