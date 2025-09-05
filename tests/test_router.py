"""
Tests for the main router module.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from system_prompt_router.router import SystemPromptRouter
from system_prompt_router.config import Config


class TestSystemPromptRouter:
    """Test cases for SystemPromptRouter class."""
    
    @pytest.mark.unit
    def test_initialization_with_config(self, test_config, mock_openai_client):
        """Test router initialization with config."""
        with patch('system_prompt_router.router.OpenAI', return_value=mock_openai_client):
            router = SystemPromptRouter(test_config)
            
            assert router.config == test_config
            assert router.library is not None
            assert router.embedding_engine is not None
            assert router.similarity_calculator is not None
    
    @pytest.mark.unit
    def test_initialization_without_config(self, mock_openai_api_key, mock_openai_client):
        """Test router initialization without config (uses defaults)."""
        with patch('system_prompt_router.router.OpenAI', return_value=mock_openai_client):
            router = SystemPromptRouter()
            
            assert router.config is not None
            assert router.config.openai_api_key == "test-api-key"
    
    @pytest.mark.unit
    def test_initialization_with_overrides(self, test_config, mock_openai_client):
        """Test router initialization with parameter overrides."""
        with patch('system_prompt_router.router.OpenAI', return_value=mock_openai_client):
            router = SystemPromptRouter(
                config=test_config,
                embedding_model="custom-model",
                openai_model="custom-openai",
                openai_api_key="custom-key"
            )
            
            assert router.config.get("embedding_model") == "custom-model"
            assert router.config.get("openai_model") == "custom-openai"
            assert router.config.get("openai_api_key") == "custom-key"
    
    @pytest.mark.unit
    def test_add_prompt(self, system_router):
        """Test adding a single prompt."""
        system_router.add_prompt(
            name="test_prompt",
            description="A test prompt",
            system_prompt="You are a test assistant."
        )
        
        assert len(system_router.library) == 1
        assert "test_prompt" in system_router.library
        assert not system_router._embeddings_computed
    
    @pytest.mark.unit
    def test_load_prompt_library_dict(self, system_router, sample_prompts):
        """Test loading prompt library from dictionary."""
        system_router.load_prompt_library(sample_prompts)
        
        assert len(system_router.library) == 3
        assert "test_prompt_1" in system_router.library
        assert not system_router._embeddings_computed
    
    @pytest.mark.unit
    def test_load_prompt_library_file(self, system_router, test_prompts_file):
        """Test loading prompt library from file."""
        system_router.load_prompt_library(str(test_prompts_file))
        
        assert len(system_router.library) == 3
        assert "test_prompt_1" in system_router.library
    
    @pytest.mark.unit
    def test_load_prompt_library_invalid_file(self, system_router):
        """Test loading from invalid file format."""
        with pytest.raises(ValueError, match="File must be JSON or YAML format"):
            system_router.load_prompt_library("test.txt")
    
    @pytest.mark.unit
    @patch('system_prompt_router.router.SystemPromptRouter._compute_prompt_embeddings')
    def test_find_best_prompt(self, mock_compute, system_router, sample_prompts):
        """Test finding best matching prompts."""
        # Setup
        system_router.load_prompt_library(sample_prompts)
        system_router._embeddings_computed = True
        system_router._prompt_embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        
        # Mock embedding engine
        mock_embedding = np.array([0.2, 0.3, 0.4])
        system_router.embedding_engine.encode_text = Mock(return_value=mock_embedding)
        
        # Mock similarity calculator
        mock_similarities = [
            ("test_prompt_1", 0.9),
            ("test_prompt_2", 0.7),
            ("test_prompt_3", 0.5)
        ]
        system_router.similarity_calculator.find_top_k_similar = Mock(return_value=mock_similarities)
        
        # Test
        results = system_router.find_best_prompt("test query", top_k=2)
        
        assert len(results) == 3  # Returns all mock results
        assert results[0][0] == "test_prompt_1"
        assert results[0][1] == 0.9
        assert results[0][2] == "You are a helpful coding assistant."
    
    @pytest.mark.unit
    def test_find_best_prompt_empty_library(self, system_router):
        """Test finding best prompt with empty library."""
        with pytest.raises(ValueError, match="No prompts loaded"):
            system_router.find_best_prompt("test query")
    
    @pytest.mark.unit
    def test_find_best_prompt_empty_query(self, system_router, sample_prompts):
        """Test finding best prompt with empty query."""
        system_router.load_prompt_library(sample_prompts)
        
        with pytest.raises(ValueError, match="User query cannot be empty"):
            system_router.find_best_prompt("")
    
    @pytest.mark.unit
    def test_generate_response_with_best_prompt(self, system_router, sample_prompts, mock_openai_client):
        """Test generating response using best matching prompt."""
        # Setup
        system_router.load_prompt_library(sample_prompts)
        
        # Mock find_best_prompt
        mock_matches = [("test_prompt_1", 0.9, "You are a helpful coding assistant.")]
        system_router.find_best_prompt = Mock(return_value=mock_matches)
        
        # Test
        result = system_router.generate_response("test query")
        
        assert result["response"] == "Test response"
        assert result["matched_prompt"] == "test_prompt_1"
        assert result["similarity_score"] == 0.9
        assert result["system_prompt"] == "You are a helpful coding assistant."
        assert result["user_query"] == "test query"
    
    @pytest.mark.unit
    def test_generate_response_with_custom_prompt(self, system_router, mock_openai_client):
        """Test generating response with custom system prompt."""
        custom_prompt = "You are a custom assistant."
        
        result = system_router.generate_response(
            "test query",
            use_best_prompt=False,
            custom_system_prompt=custom_prompt
        )
        
        assert result["response"] == "Test response"
        assert result["matched_prompt"] == "custom"
        assert result["similarity_score"] is None
        assert result["system_prompt"] == custom_prompt
    
    @pytest.mark.unit
    def test_generate_response_validation(self, system_router):
        """Test response generation validation."""
        # Empty query
        with pytest.raises(ValueError, match="User query cannot be empty"):
            system_router.generate_response("")
        
        # No prompts and no custom prompt
        with pytest.raises(ValueError, match="No prompts loaded"):
            system_router.generate_response("test")
        
        # Neither best prompt nor custom prompt
        with pytest.raises(ValueError, match="Either use_best_prompt must be True"):
            system_router.generate_response("test", use_best_prompt=False)
    
    @pytest.mark.unit
    def test_generate_response_openai_error(self, system_router, sample_prompts):
        """Test handling OpenAI API errors."""
        system_router.load_prompt_library(sample_prompts)
        
        # Mock find_best_prompt
        mock_matches = [("test_prompt_1", 0.9, "You are a helpful coding assistant.")]
        system_router.find_best_prompt = Mock(return_value=mock_matches)
        
        # Mock OpenAI client to raise error
        system_router.openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        with pytest.raises(RuntimeError, match="Failed to generate response"):
            system_router.generate_response("test query")
    
    @pytest.mark.unit
    def test_list_prompts(self, system_router, sample_prompts):
        """Test listing all prompts."""
        system_router.load_prompt_library(sample_prompts)
        
        prompts = system_router.list_prompts()
        
        assert len(prompts) == 3
        assert "test_prompt_1" in prompts
        assert prompts["test_prompt_1"] == "A test prompt for coding assistance"
    
    @pytest.mark.unit
    def test_get_prompt_details(self, system_router, sample_prompts):
        """Test getting prompt details."""
        system_router.load_prompt_library(sample_prompts)
        
        details = system_router.get_prompt_details("test_prompt_1")
        
        assert details is not None
        assert details["description"] == "A test prompt for coding assistance"
        assert details["system_prompt"] == "You are a helpful coding assistant."
        
        # Non-existent prompt
        assert system_router.get_prompt_details("nonexistent") is None
    
    @pytest.mark.unit
    def test_remove_prompt(self, system_router, sample_prompts):
        """Test removing a prompt."""
        system_router.load_prompt_library(sample_prompts)
        assert len(system_router.library) == 3
        
        # Remove existing prompt
        result = system_router.remove_prompt("test_prompt_1")
        assert result is True
        assert len(system_router.library) == 2
        assert not system_router._embeddings_computed
        
        # Remove non-existent prompt
        result = system_router.remove_prompt("nonexistent")
        assert result is False
    
    @pytest.mark.unit
    def test_clear_prompts(self, system_router, sample_prompts):
        """Test clearing all prompts."""
        system_router.load_prompt_library(sample_prompts)
        assert len(system_router.library) == 3
        
        system_router.clear_prompts()
        
        assert len(system_router.library) == 0
        assert not system_router._embeddings_computed
        assert system_router._prompt_embeddings is None
    
    @pytest.mark.unit
    def test_get_stats(self, system_router, sample_prompts):
        """Test getting router statistics."""
        system_router.load_prompt_library(sample_prompts)
        
        stats = system_router.get_stats()
        
        assert "library" in stats
        assert "embeddings" in stats
        assert "config" in stats
        assert "cache_status" in stats
        
        assert stats["library"]["total_prompts"] == 3
        assert stats["config"]["embedding_model"] == system_router.config.embedding_model
    
    @pytest.mark.unit
    @patch('system_prompt_router.router.SystemPromptRouter._compute_prompt_embeddings')
    def test_compare_similarity_methods(self, mock_compute, system_router, sample_prompts):
        """Test comparing similarity methods."""
        # Setup
        system_router.load_prompt_library(sample_prompts)
        system_router._embeddings_computed = True
        system_router._prompt_embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        
        # Mock embedding engine
        mock_embedding = np.array([0.2, 0.3, 0.4])
        system_router.embedding_engine.encode_text = Mock(return_value=mock_embedding)
        
        # Mock similarity calculator
        mock_comparison = {
            "cosine": [("test_prompt_1", 0.9)],
            "dot_product": [("test_prompt_2", 0.8)],
            "euclidean": [("test_prompt_3", 0.1)]
        }
        system_router.similarity_calculator.compare_methods = Mock(return_value=mock_comparison)
        
        # Test
        results = system_router.compare_similarity_methods("test query", top_k=1)
        
        assert "cosine" in results
        assert "dot_product" in results
        assert "euclidean" in results
    
    @pytest.mark.unit
    def test_validate_setup_success(self, system_router, sample_prompts):
        """Test successful setup validation."""
        system_router.load_prompt_library(sample_prompts)
        
        # Mock embedding engine to work properly
        system_router.embedding_engine.encode_text = Mock(return_value=np.array([0.1, 0.2, 0.3]))
        
        errors = system_router.validate_setup()
        
        assert len(errors) == 0
    
    @pytest.mark.unit
    def test_validate_setup_errors(self, mock_openai_client):
        """Test setup validation with errors."""
        # Create router without API key
        with patch.dict('os.environ', {}, clear=True):
            with patch('system_prompt_router.router.OpenAI', return_value=mock_openai_client):
                try:
                    router = SystemPromptRouter()
                except ValueError:
                    # Expected due to missing API key
                    pass
        
        # Create router with API key but no prompts
        with patch('system_prompt_router.config.Config._validate_config'):
            config = Config()
            config.set('openai_api_key', 'test-key')
            
            with patch('system_prompt_router.router.OpenAI', return_value=mock_openai_client):
                router = SystemPromptRouter(config)
                
                errors = router.validate_setup()
                
                assert len(errors) > 0
                assert any("No prompts loaded" in error for error in errors)
    
    @pytest.mark.unit
    @patch('system_prompt_router.router.SystemPromptRouter._compute_prompt_embeddings')
    def test_compute_prompt_embeddings(self, mock_compute, system_router, sample_prompts):
        """Test prompt embeddings computation."""
        system_router.load_prompt_library(sample_prompts)
        
        # Mock embedding engine
        mock_embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        system_router.embedding_engine.encode_texts = Mock(return_value=mock_embeddings)
        
        # Call the real method
        mock_compute.side_effect = system_router._compute_prompt_embeddings.__wrapped__
        system_router._compute_prompt_embeddings()
        
        assert system_router._embeddings_computed is True
        assert system_router._prompt_embeddings is not None
    
    @pytest.mark.unit
    def test_compute_prompt_embeddings_empty_library(self, system_router):
        """Test embeddings computation with empty library."""
        with pytest.raises(ValueError, match="No prompts loaded"):
            system_router._compute_prompt_embeddings()

