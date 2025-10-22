"""
Utility functions for logging, data saving/loading, and configuration management
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml


class AnalysisLogger:
    """Professional logging system with file and console output"""

    def __init__(self, log_dir: str = "data/processed/logs", name: str = "analysis"):
        """
        Initialize logger with file and console handlers

        Args:
            log_dir: Directory to save log files
            name: Name of the logger
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create unique log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{name}_{timestamp}.log"

        # Configure logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Remove existing handlers to avoid duplicates
        self.logger.handlers = []

        # File handler (detailed logs)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        # Console handler (important logs only)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        console_handler.setFormatter(console_formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.info(f"Log file created: {log_file}")

    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)


class DataManager:
    """Manage data saving and loading with metadata tracking"""

    def __init__(self, base_dir: str = "data/processed"):
        """
        Initialize data manager

        Args:
            base_dir: Base directory for processed data
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        sequences: np.ndarray,
        metadata: Optional[Dict[str, np.ndarray]] = None,
        filename: str = "embeddings.npz"
    ) -> Path:
        """
        Save embeddings with metadata

        Args:
            embeddings: Embedding vectors (N, D)
            labels: Protein labels (N,)
            sequences: Protein sequences (N,)
            metadata: Additional metadata (optional)
            filename: Output filename

        Returns:
            Path to saved file
        """
        filepath = self.base_dir / filename

        # Prepare data dictionary
        data_dict = {
            'embeddings': embeddings,
            'labels': labels,
            'sequences': sequences,
            'timestamp': np.array([datetime.now().isoformat()]),
            'embedding_dim': np.array([embeddings.shape[1]]),
            'num_proteins': np.array([len(labels)])
        }

        # Add optional metadata
        if metadata:
            for key, value in metadata.items():
                data_dict[key] = value

        # Save compressed
        np.savez_compressed(filepath, **data_dict)

        return filepath

    def load_embeddings(self, filename: str = "embeddings.npz") -> Dict[str, np.ndarray]:
        """
        Load embeddings with metadata

        Args:
            filename: Input filename

        Returns:
            Dictionary with all saved data
        """
        filepath = self.base_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Embeddings file not found: {filepath}")

        data = np.load(filepath, allow_pickle=True)
        return {key: data[key] for key in data.files}

    def save_attentions(
        self,
        attentions: np.ndarray,
        labels: np.ndarray,
        layer_info: Optional[Dict[str, Any]] = None,
        filename: str = "attentions.npz"
    ) -> Path:
        """
        Save attention weights

        Args:
            attentions: Attention weights (N, num_heads, seq_len, seq_len)
            labels: Protein labels (N,)
            layer_info: Information about layers/heads
            filename: Output filename

        Returns:
            Path to saved file
        """
        filepath = self.base_dir / filename

        data_dict = {
            'attentions': attentions,
            'labels': labels,
            'timestamp': np.array([datetime.now().isoformat()]),
            'attention_shape': np.array(attentions.shape)
        }

        if layer_info:
            data_dict['layer_info'] = np.array([json.dumps(layer_info)])

        np.savez_compressed(filepath, **data_dict)

        return filepath

    def load_attentions(self, filename: str = "attentions.npz") -> Dict[str, np.ndarray]:
        """Load attention weights"""
        filepath = self.base_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Attentions file not found: {filepath}")

        data = np.load(filepath, allow_pickle=True)
        result = {key: data[key] for key in data.files}

        # Parse layer_info if present
        if 'layer_info' in result:
            result['layer_info'] = json.loads(str(result['layer_info'][0]))

        return result

    def save_predictions(
        self,
        predictions_df: pd.DataFrame,
        filename: str = "predictions.csv"
    ) -> Path:
        """
        Save predictions to CSV

        Args:
            predictions_df: DataFrame with predictions
            filename: Output filename

        Returns:
            Path to saved file
        """
        filepath = self.base_dir / filename
        predictions_df.to_csv(filepath, index=False)
        return filepath

    def load_predictions(self, filename: str = "predictions.csv") -> pd.DataFrame:
        """Load predictions from CSV"""
        filepath = self.base_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Predictions file not found: {filepath}")

        return pd.read_csv(filepath)

    def save_tsne_results(
        self,
        tsne_2d: np.ndarray,
        tsne_3d: np.ndarray,
        labels: np.ndarray,
        metadata: Optional[Dict[str, np.ndarray]] = None,
        filename: str = "tsne_results.npz"
    ) -> Path:
        """
        Save t-SNE results

        Args:
            tsne_2d: 2D t-SNE coordinates
            tsne_3d: 3D t-SNE coordinates
            labels: Protein labels
            metadata: Additional metadata
            filename: Output filename

        Returns:
            Path to saved file
        """
        filepath = self.base_dir / filename

        data_dict = {
            'tsne_2d': tsne_2d,
            'tsne_3d': tsne_3d,
            'labels': labels,
            'timestamp': np.array([datetime.now().isoformat()])
        }

        if metadata:
            for key, value in metadata.items():
                data_dict[key] = value

        np.savez_compressed(filepath, **data_dict)

        return filepath

    def load_tsne_results(self, filename: str = "tsne_results.npz") -> Dict[str, np.ndarray]:
        """Load t-SNE results"""
        filepath = self.base_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"t-SNE results not found: {filepath}")

        data = np.load(filepath, allow_pickle=True)
        return {key: data[key] for key in data.files}


class ConfigManager:
    """Manage configuration from YAML file"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize config manager

        Args:
            config_path: Path to YAML config file
        """
        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            return self.get_default_config()

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def save_config(self, config: Optional[Dict[str, Any]] = None):
        """Save configuration to YAML file"""
        config_to_save = config if config is not None else self.config

        with open(self.config_path, 'w') as f:
            yaml.dump(config_to_save, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'model': {
                'path': 'model/',
                'batch_size': 16,
                'max_length': 512,
                'device': 'auto'  # 'auto', 'cuda', or 'cpu'
            },
            'data': {
                'input_csv': 'data/master.csv',
                'processed_dir': 'data/processed'
            },
            'analysis': {
                'extract_embeddings': True,
                'extract_attentions': True,
                'run_tsne': True,
                'tsne_perplexity': 30,
                'random_seed': 42
            },
            'visualization': {
                'highlight_proteins': [],  # e.g., ['lpg1484', 'lpg0502']
                'color_by': 'localisation',  # 'localisation' or 'category'
                'opacity_proteom': 0.3,
                'opacity_known': 0.9
            },
            'output': {
                'report_name': 'legionella_analysis_report.html',
                'output_dir': 'outputs',
                'save_intermediate': True
            },
            'logging': {
                'level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
                'log_dir': 'data/processed/logs'
            }
        }

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get config value using dot notation

        Args:
            key_path: Path to config value (e.g., 'model.batch_size')
            default: Default value if key not found

        Returns:
            Config value or default
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any):
        """
        Set config value using dot notation

        Args:
            key_path: Path to config value (e.g., 'model.batch_size')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value


def format_time(seconds: float) -> str:
    """
    Format seconds to human-readable string

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_size(num_bytes: int) -> str:
    """
    Format bytes to human-readable string

    Args:
        num_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
