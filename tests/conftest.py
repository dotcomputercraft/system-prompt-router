"""
Pytest configuration and shared fixtures for System Prompt Router tests.
"""

import os
import tempfile
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from system_prompt_router.config import Config
from system_prompt_router.library import PromptLibrary
from system_prompt_router.embeddings import EmbeddingEngine
from system_prompt_router.similarity import SimilarityCalculator
from system_prompt_router.router import SystemPromptRouter


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_openai_api_key():
    """Mock OpenAI API key for testing."""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'}):
        yield 'test-api-key'


@pytest.fixture
def test_config(mock_openai_api_key, temp_dir):
    """Create a test configuration."""
    config = Config()
    config.set('cache_dir', str(temp_dir / 'cache'))
    config.set('cache_embeddings', False)  # Disable caching for tests
    return config


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return {
        "test_prompt_1": {
            "description": "A test prompt for coding assistance",
            "system_prompt": "You are a helpful coding assistant."
        },
        "test_prompt_2": {
            "description": "A test prompt for writing help",
            "system_prompt": "You are a creative writing assistant."
        },
        "test_prompt_3": {
            "description": "A test prompt for data analysis",
            "system_prompt": "You are a data analysis expert."
        }
    }


@pytest.fixture
def prompt_library(sample_prompts):
    """Create a prompt library with sample prompts."""
    library = PromptLibrary()
    library.load_from_dict(sample_prompts)
    return library


@pytest.fixture
def mock_embedding_model():
    """Mock sentence transformer model."""
    mock_model = Mock()
    mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_model.get_sentence_embedding_dimension.return_value = 3
    return mock_model


@pytest.fixture
def embedding_engine(test_config):
    """Create an embedding engine for testing."""
    return EmbeddingEngine(test_config)


@pytest.fixture
def similarity_calculator():
    """Create a similarity calculator for testing."""
    return SimilarityCalculator(similarity_threshold=0.5)


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Test response"
    mock_response.usage.total_tokens = 100
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def system_router(test_config, mock_openai_client):
    """Create a system router for testing."""
    with patch('system_prompt_router.router.OpenAI', return_value=mock_openai_client):
        router = SystemPromptRouter(test_config)
        return router


@pytest.fixture
def sample_embeddings():
    """Sample embedding vectors for testing."""
    import numpy as np
    return np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ])


@pytest.fixture
def sample_query_embedding():
    """Sample query embedding for testing."""
    import numpy as np
    return np.array([0.2, 0.3, 0.4])


@pytest.fixture
def test_prompts_file(temp_dir, sample_prompts):
    """Create a test prompts JSON file."""
    import json
    prompts_file = temp_dir / "test_prompts.json"
    with open(prompts_file, 'w') as f:
        json.dump(sample_prompts, f)
    return prompts_file


@pytest.fixture
def test_config_file(temp_dir):
    """Create a test configuration file."""
    import yaml
    config_file = temp_dir / "test_config.yaml"
    config_data = {
        'embedding_model': 'test-model',
        'openai_model': 'test-openai-model',
        'similarity_threshold': 0.6,
        'top_k_results': 5
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    return config_file


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )

