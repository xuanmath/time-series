"""
Logging utilities for time series training
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import sys


class TrainingLogger:
    """
    Logger for training experiments.
    
    Features:
    - Console logging with colors
    - File logging to logs/
    - JSON metrics logging
    - Experiment tracking
    """
    
    def __init__(
        self,
        name: str = "training",
        log_dir: str = "logs",
        experiment_name: Optional[str] = None
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate experiment name
        if experiment_name is None:
            self.experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.experiment_name = experiment_name
        
        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / f"{self.experiment_name}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Metrics storage
        self.metrics = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "model": {},
            "data": {},
            "training": {},
            "results": {}
        }
        
        self.start_datetime = datetime.now()
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def log_separator(self, char: str = "=", length: int = 60):
        """Log a separator line."""
        self.info(char * length)
    
    def log_header(self, title: str):
        """Log a header with separators."""
        self.log_separator()
        self.info(title)
        self.log_separator()
    
    def log_model_config(self, model_name: str, config: Dict[str, Any]):
        """Log model configuration."""
        self.info(f"\n[CONFIG] Model Configuration: {model_name}")
        self.metrics["model"]["name"] = model_name
        self.metrics["model"]["config"] = config
        
        for key, value in config.items():
            self.info(f"    {key}: {value}")
    
    def log_data_info(
        self,
        n_samples: int,
        n_features: int,
        n_train: int,
        n_test: int,
        time_range: Optional[str] = None
    ):
        """Log dataset information."""
        self.info("\n[DATA] Dataset Information")
        self.info(f"    Total samples: {n_samples}")
        self.info(f"    Features: {n_features}")
        self.info(f"    Train samples: {n_train}")
        self.info(f"    Test samples: {n_test}")
        if time_range:
            self.info(f"    Time range: {time_range}")
        
        self.metrics["data"] = {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_train": n_train,
            "n_test": n_test,
            "time_range": time_range
        }
    
    def log_training_start(self, model_name: str):
        """Log training start."""
        self.info(f"\n[TRAIN] Training {model_name}...")
        self.start_datetime = datetime.now()
    
    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        lr: Optional[float] = None
    ):
        """Log epoch progress."""
        msg = f"Epoch {epoch}/{total_epochs} - Train Loss: {train_loss:.6f}"
        if val_loss is not None:
            msg += f" - Val Loss: {val_loss:.6f}"
        if lr is not None:
            msg += f" - LR: {lr:.6f}"
        self.info(msg)
        
        # Store in metrics
        if "epochs" not in self.metrics["training"]:
            self.metrics["training"]["epochs"] = []
        
        self.metrics["training"]["epochs"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr
        })
    
    def log_early_stopping(self, epoch: int):
        """Log early stopping."""
        self.info(f"[WARN] Early stopping at epoch {epoch}")
        self.metrics["training"]["early_stopping"] = True
        self.metrics["training"]["stopped_epoch"] = epoch
    
    def log_training_complete(self):
        """Log training completion."""
        self.info("[OK] Training completed!")
    
    def log_results(
        self,
        metrics: Dict[str, float],
        highlight_best: bool = True
    ):
        """Log evaluation results."""
        self.info("\n[RESULTS] Evaluation Results")
        self.log_separator("-")
        
        best_metric = min(metrics.items(), key=lambda x: x[1] if "loss" in x[0] or "error" in x[0].lower() or "mae" in x[0].lower() or "rmse" in x[0].lower() else float('inf'))
        
        for name, value in metrics.items():
            marker = " *" if highlight_best and name == best_metric[0] else ""
            if isinstance(value, float):
                self.info(f"    {name.upper():>12}: {value:.4f}{marker}")
            else:
                self.info(f"    {name.upper():>12}: {value}{marker}")
        
        self.metrics["results"] = metrics
    
    def log_feature_importance(
        self,
        importance: Dict[str, float],
        top_k: int = 10
    ):
        """Log feature importance."""
        self.info(f"\n[FEATURES] Top {top_k} Features")
        
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        for i, (name, value) in enumerate(sorted_items, 1):
            self.info(f"    {i:2d}. {name}: {value:.4f}")
        
        self.metrics["results"]["feature_importance"] = dict(sorted_items)
    
    def log_file_saved(self, file_type: str, filepath: str):
        """Log file saved."""
        self.info(f"[SAVE] {file_type} saved to: {filepath}")
    
    def finalize(self) -> Dict[str, Any]:
        """Finalize logging and return metrics."""
        end_datetime = datetime.now()
        duration = (end_datetime - self.start_datetime).total_seconds()
        
        self.metrics["end_time"] = end_datetime.isoformat()
        self.metrics["duration_seconds"] = duration
        
        self.info(f"\n[TIME] Total duration: {duration:.2f} seconds")
        
        # Save metrics to JSON
        metrics_file = self.log_dir / f"{self.experiment_name}_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        self.info(f"[SAVE] Metrics saved to: {metrics_file}")
        
        return self.metrics


def setup_experiment(
    model_name: str,
    config: Dict[str, Any],
    log_dir: str = "logs"
) -> TrainingLogger:
    """
    Setup experiment logging.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    config : Dict[str, Any]
        Model configuration
    log_dir : str
        Directory for logs
        
    Returns
    -------
    logger : TrainingLogger
        Configured logger instance
    """
    experiment_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = TrainingLogger(name=model_name, log_dir=log_dir, experiment_name=experiment_name)
    
    logger.log_header(f"Experiment: {experiment_name}")
    logger.log_model_config(model_name, config)
    
    return logger
