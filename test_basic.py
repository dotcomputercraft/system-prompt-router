#!/usr/bin/env python3
"""
Basic test script for System Prompt Router without heavy dependencies.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test basic imports."""
    try:
        from system_prompt_router.config import Config
        from system_prompt_router.library import PromptLibrary
        from system_prompt_router.similarity import SimilarityCalculator
        
        print("✓ All imports successful")
        return True, Config, PromptLibrary, SimilarityCalculator
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False, None, None, None

def test_config(Config):
    """Test configuration."""
    try:
        # Mock environment
        os.environ['OPENAI_API_KEY'] = 'test-key'
        
        config = Config()
        print(f"✓ Config created with model: {config.embedding_model}")
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_prompt_library(PromptLibrary):
    """Test prompt library."""
    try:
        library = PromptLibrary()
        
        # Add a test prompt
        library.add_prompt(
            "test_prompt",
            "A test prompt for demonstration",
            "You are a helpful assistant."
        )
        
        print(f"✓ PromptLibrary working, {len(library)} prompts loaded")
        return True
    except Exception as e:
        print(f"✗ PromptLibrary test failed: {e}")
        return False

def test_similarity(SimilarityCalculator):
    """Test similarity calculator."""
    try:
        import numpy as np
        
        calc = SimilarityCalculator()
        
        # Test with simple vectors
        query = np.array([1.0, 0.0, 0.0])
        prompts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        
        similarities = calc.calculate_cosine_similarity(query, prompts)
        print(f"✓ SimilarityCalculator working, similarities: {similarities}")
        return True
    except Exception as e:
        print(f"✗ SimilarityCalculator test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== System Prompt Router Basic Tests ===\n")
    
    # Test imports first
    print("Testing Imports:")
    import_success, Config, PromptLibrary, SimilarityCalculator = test_imports()
    
    if not import_success:
        print("=== Results: 0/4 tests passed ===")
        print("❌ Import tests failed")
        return 1
    
    tests = [
        ("Config", lambda: test_config(Config)),
        ("PromptLibrary", lambda: test_prompt_library(PromptLibrary)),
        ("SimilarityCalculator", lambda: test_similarity(SimilarityCalculator)),
    ]
    
    passed = 1  # Imports already passed
    total = len(tests) + 1
    
    for name, test_func in tests:
        print(f"\nTesting {name}:")
        if test_func():
            passed += 1
    
    print(f"\n=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
