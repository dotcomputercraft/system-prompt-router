"""
Tests for the configuration module.
"""

import os
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

from system_prompt_router.config import Config


class TestConfig:
    """Test cases for Config class."""
    
    @pytest.mark.unit
    def test_default_config(self, mock_openai_api_key):
        """Test default configuration values."""
        config = Config()
        
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.openai_model == "gpt-3.5-turbo"
        assert config.openai_api_key == "test-api-key"
        assert config.similarity_threshold == 0.5
        assert config.top_k_results == 3
        assert config.cache_embeddings is True
    
    @pytest.mark.unit
    def test_config_from_env_vars(self):
        """Test configuration loading from environment variables."""
        env_vars = {
            'OPENAI_API_KEY': 'env-api-key',
            'EMBEDDING_MODEL': 'custom-model',
            'OPENAI_MODEL': 'gpt-4',
            'SIMILARITY_THRESHOLD': '0.7',
            'TOP_K_RESULTS': '5',
            'CACHE_EMBEDDINGS': 'false',
            'TEMPERATURE': '0.9'
        }
        
        with patch.dict(os.environ, env_vars):
            config = Config()
            
            assert config.openai_api_key == 'env-api-key'
            assert config.embedding_model == 'custom-model'
            assert config.openai_model == 'gpt-4'
            assert config.similarity_threshold == 0.7
            assert config.top_k_results == 5
            assert config.cache_embeddings is False
            assert config.get('temperature') == 0.9
    
    @pytest.mark.unit
    def test_config_from_file(self, temp_dir, mock_openai_api_key):
        """Test configuration loading from YAML file."""
        config_file = temp_dir / "config.yaml"
        config_data = """
        embedding_model: file-model
        openai_model: file-openai-model
        similarity_threshold: 0.8
        top_k_results: 7
        """
        
        with open(config_file, 'w') as f:
            f.write(config_data)
        
        config = Config(str(config_file))
        
        assert config.embedding_model == 'file-model'
        assert config.openai_model == 'file-openai-model'
        assert config.similarity_threshold == 0.8
        assert config.top_k_results == 7
    
    @pytest.mark.unit
    def test_env_vars_override_file(self, temp_dir):
        """Test that environment variables override file configuration."""
        config_file = temp_dir / "config.yaml"
        config_data = """
        embedding_model: file-model
        openai_model: file-openai-model
        """
        
        with open(config_file, 'w') as f:
            f.write(config_data)
        
        env_vars = {
            'OPENAI_API_KEY': 'env-api-key',
            'EMBEDDING_MODEL': 'env-model'
        }
        
        with patch.dict(os.environ, env_vars):
            config = Config(str(config_file))
            
            assert config.embedding_model == 'env-model'  # From env
            assert config.openai_model == 'file-openai-model'  # From file
    
    @pytest.mark.unit
    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                Config()
    
    @pytest.mark.unit
    def test_invalid_similarity_threshold_raises_error(self, mock_openai_api_key):
        """Test that invalid similarity threshold raises ValueError."""
        with patch.dict(os.environ, {'SIMILARITY_THRESHOLD': '1.5'}):
            with pytest.raises(ValueError, match="similarity_threshold must be between 0.0 and 1.0"):
                Config()
    
    @pytest.mark.unit
    def test_invalid_temperature_raises_error(self, mock_openai_api_key):
        """Test that invalid temperature raises ValueError."""
        with patch.dict(os.environ, {'TEMPERATURE': '3.0'}):
            with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
                Config()
    
    @pytest.mark.unit
    def test_invalid_top_k_raises_error(self, mock_openai_api_key):
        """Test that invalid top_k raises ValueError."""
        with patch.dict(os.environ, {'TOP_K_RESULTS': '0'}):
            with pytest.raises(ValueError, match="top_k_results must be at least 1"):
                Config()
    
    @pytest.mark.unit
    def test_get_and_set_methods(self, test_config):
        """Test get and set methods."""
        # Test get with default
        assert test_config.get('nonexistent_key', 'default') == 'default'
        
        # Test set and get
        test_config.set('custom_key', 'custom_value')
        assert test_config.get('custom_key') == 'custom_value'
    
    @pytest.mark.unit
    def test_get_all_config(self, test_config):
        """Test get_all method."""
        all_config = test_config.get_all()
        
        assert isinstance(all_config, dict)
        assert 'embedding_model' in all_config
        assert 'openai_model' in all_config
        assert 'openai_api_key' in all_config
    
    @pytest.mark.unit
    def test_save_to_file(self, test_config, temp_dir):
        """Test saving configuration to file."""
        config_file = temp_dir / "saved_config.yaml"
        test_config.save_to_file(str(config_file))
        
        assert config_file.exists()
        
        # Load and verify
        new_config = Config(str(config_file))
        assert new_config.embedding_model == test_config.embedding_model
        assert new_config.openai_model == test_config.openai_model
    
    @pytest.mark.unit
    def test_config_properties(self, test_config):
        """Test configuration properties."""
        assert isinstance(test_config.embedding_model, str)
        assert isinstance(test_config.openai_model, str)
        assert isinstance(test_config.openai_api_key, str)
        assert isinstance(test_config.similarity_threshold, float)
        assert isinstance(test_config.top_k_results, int)
        assert isinstance(test_config.cache_embeddings, bool)
        assert isinstance(test_config.cache_dir, str)
    
    @pytest.mark.unit
    def test_nonexistent_config_file(self, mock_openai_api_key):
        """Test handling of nonexistent config file."""
        # Should not raise error, just use defaults
        config = Config("nonexistent_file.yaml")
        assert config.embedding_model == "all-MiniLM-L6-v2"

