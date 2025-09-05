"""
Configuration management for System Prompt Router.

This module handles loading and managing configuration settings from environment
variables, configuration files, and default values.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """Configuration manager for System Prompt Router."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file (optional)
        """
        # Load environment variables
        load_dotenv()
        
        # Default configuration
        self._config = {
            "embedding_model": "all-MiniLM-L6-v2",
            "openai_model": "gpt-3.5-turbo",
            "openai_api_key": None,
            "openai_api_base": None,
            "similarity_threshold": 0.5,
            "top_k_results": 3,
            "cache_embeddings": True,
            "cache_dir": ".cache",
            "log_level": "INFO",
            "max_tokens": 1000,
            "temperature": 0.7,
        }
        
        # Load from config file if provided
        if config_file:
            self.load_config_file(config_file)
        
        # Override with environment variables
        self._load_from_env()
        
        # Validate configuration
        self._validate_config()
    
    def load_config_file(self, config_file: str) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_file: Path to YAML configuration file
        """
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    self._config.update(file_config)
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            "EMBEDDING_MODEL": "embedding_model",
            "OPENAI_MODEL": "openai_model", 
            "OPENAI_API_KEY": "openai_api_key",
            "OPENAI_API_BASE": "openai_api_base",
            "SIMILARITY_THRESHOLD": "similarity_threshold",
            "TOP_K_RESULTS": "top_k_results",
            "CACHE_EMBEDDINGS": "cache_embeddings",
            "CACHE_DIR": "cache_dir",
            "LOG_LEVEL": "log_level",
            "MAX_TOKENS": "max_tokens",
            "TEMPERATURE": "temperature",
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if config_key in ["similarity_threshold", "temperature"]:
                    self._config[config_key] = float(value)
                elif config_key in ["top_k_results", "max_tokens"]:
                    self._config[config_key] = int(value)
                elif config_key == "cache_embeddings":
                    self._config[config_key] = value.lower() in ["true", "1", "yes"]
                else:
                    self._config[config_key] = value
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Check required API key
        if not self._config["openai_api_key"]:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or provide it in configuration file."
            )
        
        # Validate numeric ranges
        if not 0.0 <= self._config["similarity_threshold"] <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self._config["temperature"] <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        
        if self._config["top_k_results"] < 1:
            raise ValueError("top_k_results must be at least 1")
        
        if self._config["max_tokens"] < 1:
            raise ValueError("max_tokens must be at least 1")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config[key] = value
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            Dictionary of all configuration values
        """
        return self._config.copy()
    
    def save_to_file(self, config_file: str) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            config_file: Path to save configuration file
        """
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    @property
    def embedding_model(self) -> str:
        """Get embedding model name."""
        return self._config["embedding_model"]
    
    @property
    def openai_model(self) -> str:
        """Get OpenAI model name."""
        return self._config["openai_model"]
    
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key."""
        return self._config["openai_api_key"]
    
    @property
    def openai_api_base(self) -> Optional[str]:
        """Get OpenAI API base URL."""
        return self._config["openai_api_base"]
    
    @property
    def similarity_threshold(self) -> float:
        """Get similarity threshold."""
        return self._config["similarity_threshold"]
    
    @property
    def top_k_results(self) -> int:
        """Get number of top results to return."""
        return self._config["top_k_results"]
    
    @property
    def cache_embeddings(self) -> bool:
        """Get whether to cache embeddings."""
        return self._config["cache_embeddings"]
    
    @property
    def cache_dir(self) -> str:
        """Get cache directory path."""
        return self._config["cache_dir"]

