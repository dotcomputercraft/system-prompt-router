"""
Integration tests for System Prompt Router.

These tests verify the complete workflow from query to response generation.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from system_prompt_router.router import SystemPromptRouter
from system_prompt_router.config import Config


class TestIntegration:
    """Integration test cases for complete workflows."""
    
    @pytest.mark.integration
    def test_complete_workflow_with_mocked_models(self, temp_dir, sample_prompts):
        """Test complete workflow with mocked external dependencies."""
        # Create config
        config = Config()
        config.set('cache_dir', str(temp_dir / 'cache'))
        config.set('cache_embeddings', False)
        
        # Mock OpenAI client
        mock_openai_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a helpful response about coding."
        mock_response.usage.total_tokens = 150
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Mock sentence transformer
        mock_model = Mock()
        # Query embedding
        mock_model.encode.side_effect = [
            np.array([0.5, 0.5, 0.5]),  # Query embedding
            np.array([  # Prompt embeddings
                [0.9, 0.1, 0.1],  # Close to query
                [0.1, 0.9, 0.1],  # Distant from query
                [0.1, 0.1, 0.9],  # Distant from query
            ])
        ]
        mock_model.get_sentence_embedding_dimension.return_value = 3
        
        with patch('system_prompt_router.router.OpenAI', return_value=mock_openai_client), \
             patch('system_prompt_router.embeddings.SentenceTransformer', return_value=mock_model):
            
            # Initialize router
            router = SystemPromptRouter(config)
            
            # Load prompts
            router.load_prompt_library(sample_prompts)
            
            # Process query
            query = "Help me write a Python function"
            response = router.generate_response(query)
            
            # Verify response
            assert response["response"] == "This is a helpful response about coding."
            assert response["matched_prompt"] in sample_prompts
            assert response["similarity_score"] is not None
            assert response["user_query"] == query
            assert response["tokens_used"] == 150
    
    @pytest.mark.integration
    def test_batch_processing_workflow(self, temp_dir, sample_prompts):
        """Test processing multiple queries in sequence."""
        config = Config()
        config.set('cache_dir', str(temp_dir / 'cache'))
        config.set('cache_embeddings', True)  # Test caching
        
        # Mock dependencies
        mock_openai_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage.total_tokens = 100
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        mock_model = Mock()
        mock_model.encode.side_effect = [
            np.array([0.5, 0.5, 0.5]),  # First query
            np.array([  # Prompt embeddings (computed once)
                [0.9, 0.1, 0.1],
                [0.1, 0.9, 0.1],
                [0.1, 0.1, 0.9],
            ]),
            np.array([0.6, 0.4, 0.3]),  # Second query
            np.array([0.3, 0.7, 0.2]),  # Third query
        ]
        mock_model.get_sentence_embedding_dimension.return_value = 3
        
        with patch('system_prompt_router.router.OpenAI', return_value=mock_openai_client), \
             patch('system_prompt_router.embeddings.SentenceTransformer', return_value=mock_model):
            
            router = SystemPromptRouter(config)
            router.load_prompt_library(sample_prompts)
            
            queries = [
                "Help with coding",
                "Write a story",
                "Analyze data"
            ]
            
            responses = []
            for query in queries:
                response = router.generate_response(query)
                responses.append(response)
            
            # Verify all responses
            assert len(responses) == 3
            for response in responses:
                assert response["response"] == "Response"
                assert response["matched_prompt"] in sample_prompts
                assert response["similarity_score"] is not None
    
    @pytest.mark.integration
    def test_prompt_library_management_workflow(self, temp_dir):
        """Test complete prompt library management workflow."""
        config = Config()
        config.set('cache_dir', str(temp_dir / 'cache'))
        
        mock_openai_client = Mock()
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_model.get_sentence_embedding_dimension.return_value = 3
        
        with patch('system_prompt_router.router.OpenAI', return_value=mock_openai_client), \
             patch('system_prompt_router.embeddings.SentenceTransformer', return_value=mock_model):
            
            router = SystemPromptRouter(config)
            
            # Start with empty library
            assert len(router.library) == 0
            
            # Add prompts individually
            router.add_prompt(
                "coder", 
                "Help with programming tasks",
                "You are a programming assistant."
            )
            router.add_prompt(
                "writer",
                "Help with writing tasks", 
                "You are a writing assistant."
            )
            
            assert len(router.library) == 2
            
            # List prompts
            prompts = router.list_prompts()
            assert "coder" in prompts
            assert "writer" in prompts
            
            # Get prompt details
            details = router.get_prompt_details("coder")
            assert details["description"] == "Help with programming tasks"
            
            # Remove a prompt
            removed = router.remove_prompt("writer")
            assert removed is True
            assert len(router.library) == 1
            
            # Clear all prompts
            router.clear_prompts()
            assert len(router.library) == 0
    
    @pytest.mark.integration
    def test_configuration_workflow(self, temp_dir):
        """Test configuration loading and management workflow."""
        # Create config file
        config_file = temp_dir / "test_config.yaml"
        config_content = """
        embedding_model: test-embedding-model
        openai_model: test-openai-model
        similarity_threshold: 0.8
        top_k_results: 5
        cache_embeddings: false
        """
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        # Create environment file
        env_file = temp_dir / ".env"
        with open(env_file, 'w') as f:
            f.write("OPENAI_API_KEY=test-env-key\n")
            f.write("TOP_K_RESULTS=7\n")  # Should override config file
        
        # Mock dotenv to load our test env file
        with patch('system_prompt_router.config.load_dotenv') as mock_load_dotenv:
            def mock_load_env():
                import os
                os.environ['OPENAI_API_KEY'] = 'test-env-key'
                os.environ['TOP_K_RESULTS'] = '7'
            
            mock_load_dotenv.side_effect = mock_load_env
            
            # Load config
            config = Config(str(config_file))
            
            # Verify config values
            assert config.embedding_model == "test-embedding-model"  # From file
            assert config.openai_model == "test-openai-model"  # From file
            assert config.openai_api_key == "test-env-key"  # From env
            assert config.similarity_threshold == 0.8  # From file
            assert config.top_k_results == 7  # From env (overrides file)
            assert config.cache_embeddings is False  # From file
    
    @pytest.mark.integration
    def test_similarity_methods_comparison_workflow(self, temp_dir, sample_prompts):
        """Test comparing different similarity methods."""
        config = Config()
        config.set('cache_dir', str(temp_dir / 'cache'))
        
        mock_openai_client = Mock()
        mock_model = Mock()
        
        # Mock embeddings for similarity comparison
        query_embedding = np.array([1.0, 0.0, 0.0])
        prompt_embeddings = np.array([
            [1.0, 0.0, 0.0],  # Perfect match
            [0.0, 1.0, 0.0],  # Orthogonal
            [0.5, 0.5, 0.0],  # Partial match
        ])
        
        mock_model.encode.side_effect = [query_embedding, prompt_embeddings]
        mock_model.get_sentence_embedding_dimension.return_value = 3
        
        with patch('system_prompt_router.router.OpenAI', return_value=mock_openai_client), \
             patch('system_prompt_router.embeddings.SentenceTransformer', return_value=mock_model):
            
            router = SystemPromptRouter(config)
            router.load_prompt_library(sample_prompts)
            
            # Compare similarity methods
            comparison = router.compare_similarity_methods("test query", top_k=2)
            
            # Verify comparison results
            assert "cosine" in comparison
            assert "dot_product" in comparison
            assert "euclidean" in comparison
            
            # Each method should return results
            for method, results in comparison.items():
                assert len(results) == 2  # top_k=2
                assert all(isinstance(result, tuple) and len(result) == 2 for result in results)
    
    @pytest.mark.integration
    def test_error_handling_workflow(self, temp_dir, sample_prompts):
        """Test error handling in various scenarios."""
        config = Config()
        config.set('cache_dir', str(temp_dir / 'cache'))
        
        mock_openai_client = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 3
        
        with patch('system_prompt_router.router.OpenAI', return_value=mock_openai_client), \
             patch('system_prompt_router.embeddings.SentenceTransformer', return_value=mock_model):
            
            router = SystemPromptRouter(config)
            
            # Test with no prompts loaded
            with pytest.raises(ValueError, match="No prompts loaded"):
                router.find_best_prompt("test query")
            
            # Load prompts
            router.load_prompt_library(sample_prompts)
            
            # Test with empty query
            with pytest.raises(ValueError, match="cannot be empty"):
                router.find_best_prompt("")
            
            # Test OpenAI API error
            mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
            
            with pytest.raises(RuntimeError, match="Failed to generate response"):
                router.generate_response("test query")
    
    @pytest.mark.integration
    def test_validation_workflow(self, temp_dir, sample_prompts):
        """Test setup validation workflow."""
        config = Config()
        config.set('cache_dir', str(temp_dir / 'cache'))
        
        mock_openai_client = Mock()
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_model.get_sentence_embedding_dimension.return_value = 3
        
        with patch('system_prompt_router.router.OpenAI', return_value=mock_openai_client), \
             patch('system_prompt_router.embeddings.SentenceTransformer', return_value=mock_model):
            
            router = SystemPromptRouter(config)
            
            # Validate without prompts (should have errors)
            errors = router.validate_setup()
            assert len(errors) > 0
            assert any("No prompts loaded" in error for error in errors)
            
            # Load prompts and validate again
            router.load_prompt_library(sample_prompts)
            errors = router.validate_setup()
            
            # Should have no errors now
            assert len(errors) == 0
    
    @pytest.mark.integration
    def test_statistics_workflow(self, temp_dir, sample_prompts):
        """Test statistics collection workflow."""
        config = Config()
        config.set('cache_dir', str(temp_dir / 'cache'))
        config.set('cache_embeddings', True)
        
        mock_openai_client = Mock()
        mock_model = Mock()
        mock_model.encode.side_effect = [
            np.array([0.1, 0.2, 0.3]),  # Query embedding
            np.array([  # Prompt embeddings
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ])
        ]
        mock_model.get_sentence_embedding_dimension.return_value = 3
        
        with patch('system_prompt_router.router.OpenAI', return_value=mock_openai_client), \
             patch('system_prompt_router.embeddings.SentenceTransformer', return_value=mock_model):
            
            router = SystemPromptRouter(config)
            router.load_prompt_library(sample_prompts)
            
            # Get initial stats
            stats = router.get_stats()
            
            # Verify stats structure
            assert "library" in stats
            assert "embeddings" in stats
            assert "config" in stats
            assert "cache_status" in stats
            
            # Verify library stats
            assert stats["library"]["total_prompts"] == 3
            
            # Verify config stats
            assert stats["config"]["embedding_model"] == config.embedding_model
            assert stats["config"]["openai_model"] == config.openai_model
            
            # Process a query to generate embeddings
            router.find_best_prompt("test query")
            
            # Get updated stats
            updated_stats = router.get_stats()
            assert updated_stats["cache_status"]["embeddings_computed"] is True

