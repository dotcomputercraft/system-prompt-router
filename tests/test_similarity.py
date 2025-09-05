"""
Tests for the similarity calculator module.
"""

import numpy as np
import pytest

from system_prompt_router.similarity import SimilarityCalculator


class TestSimilarityCalculator:
    """Test cases for SimilarityCalculator class."""
    
    @pytest.mark.unit
    def test_initialization(self):
        """Test calculator initialization."""
        calc = SimilarityCalculator(similarity_threshold=0.7)
        assert calc.similarity_threshold == 0.7
        
        # Test default threshold
        calc_default = SimilarityCalculator()
        assert calc_default.similarity_threshold == 0.5
    
    @pytest.mark.unit
    def test_cosine_similarity_calculation(self):
        """Test cosine similarity calculation."""
        calc = SimilarityCalculator()
        
        query_embedding = np.array([1.0, 0.0, 0.0])
        prompt_embeddings = np.array([
            [1.0, 0.0, 0.0],  # Identical vector (similarity = 1.0)
            [0.0, 1.0, 0.0],  # Orthogonal vector (similarity = 0.0)
            [0.5, 0.5, 0.0],  # 45-degree angle (similarity ≈ 0.707)
        ])
        
        similarities = calc.calculate_cosine_similarity(query_embedding, prompt_embeddings)
        
        assert len(similarities) == 3
        assert abs(similarities[0] - 1.0) < 1e-6  # Identical
        assert abs(similarities[1] - 0.0) < 1e-6  # Orthogonal
        assert abs(similarities[2] - 0.7071067811865476) < 1e-6  # 45 degrees
    
    @pytest.mark.unit
    def test_dot_product_similarity_calculation(self):
        """Test dot product similarity calculation."""
        calc = SimilarityCalculator()
        
        query_embedding = np.array([1.0, 2.0, 3.0])
        prompt_embeddings = np.array([
            [1.0, 1.0, 1.0],  # Dot product = 6
            [2.0, 2.0, 2.0],  # Dot product = 12
            [0.0, 0.0, 0.0],  # Dot product = 0
        ])
        
        similarities = calc.calculate_dot_product_similarity(query_embedding, prompt_embeddings)
        
        assert len(similarities) == 3
        assert similarities[0] == 6.0
        assert similarities[1] == 12.0
        assert similarities[2] == 0.0
    
    @pytest.mark.unit
    def test_euclidean_distance_calculation(self):
        """Test Euclidean distance calculation."""
        calc = SimilarityCalculator()
        
        query_embedding = np.array([0.0, 0.0, 0.0])
        prompt_embeddings = np.array([
            [0.0, 0.0, 0.0],  # Distance = 0
            [1.0, 0.0, 0.0],  # Distance = 1
            [3.0, 4.0, 0.0],  # Distance = 5
        ])
        
        distances = calc.calculate_euclidean_distance(query_embedding, prompt_embeddings)
        
        assert len(distances) == 3
        assert distances[0] == 0.0
        assert distances[1] == 1.0
        assert distances[2] == 5.0
    
    @pytest.mark.unit
    def test_invalid_embedding_dimensions(self):
        """Test error handling for invalid embedding dimensions."""
        calc = SimilarityCalculator()
        
        # Wrong query embedding dimension
        query_2d = np.array([[1.0, 2.0]])
        prompt_embeddings = np.array([[1.0, 2.0]])
        
        with pytest.raises(ValueError, match="Query embedding must be 1-dimensional"):
            calc.calculate_cosine_similarity(query_2d, prompt_embeddings)
        
        # Wrong prompt embeddings dimension
        query_1d = np.array([1.0, 2.0])
        prompt_1d = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="Prompt embeddings must be 2-dimensional"):
            calc.calculate_cosine_similarity(query_1d, prompt_1d)
        
        # Mismatched dimensions
        query_3d = np.array([1.0, 2.0, 3.0])
        prompt_2d = np.array([[1.0, 2.0]])
        
        with pytest.raises(ValueError, match="Embedding dimensions don't match"):
            calc.calculate_cosine_similarity(query_3d, prompt_2d)
    
    @pytest.mark.unit
    def test_find_top_k_similar_cosine(self):
        """Test finding top-k similar prompts using cosine similarity."""
        calc = SimilarityCalculator()
        
        query_embedding = np.array([1.0, 0.0, 0.0])
        prompt_embeddings = np.array([
            [1.0, 0.0, 0.0],  # similarity = 1.0
            [0.0, 1.0, 0.0],  # similarity = 0.0
            [0.7071, 0.7071, 0.0],  # similarity ≈ 0.707
        ])
        prompt_names = ["prompt_a", "prompt_b", "prompt_c"]
        
        results = calc.find_top_k_similar(
            query_embedding, prompt_embeddings, prompt_names, top_k=2, method="cosine"
        )
        
        assert len(results) == 2
        assert results[0][0] == "prompt_a"  # Highest similarity
        assert results[1][0] == "prompt_c"  # Second highest
        assert results[0][1] > results[1][1]  # Scores in descending order
    
    @pytest.mark.unit
    def test_find_top_k_similar_euclidean(self):
        """Test finding top-k similar prompts using Euclidean distance."""
        calc = SimilarityCalculator()
        
        query_embedding = np.array([0.0, 0.0, 0.0])
        prompt_embeddings = np.array([
            [0.0, 0.0, 0.0],  # distance = 0
            [3.0, 4.0, 0.0],  # distance = 5
            [1.0, 0.0, 0.0],  # distance = 1
        ])
        prompt_names = ["prompt_a", "prompt_b", "prompt_c"]
        
        results = calc.find_top_k_similar(
            query_embedding, prompt_embeddings, prompt_names, top_k=2, method="euclidean"
        )
        
        assert len(results) == 2
        assert results[0][0] == "prompt_a"  # Smallest distance
        assert results[1][0] == "prompt_c"  # Second smallest
        assert results[0][1] < results[1][1]  # Distances in ascending order
    
    @pytest.mark.unit
    def test_find_top_k_validation(self):
        """Test validation in find_top_k_similar."""
        calc = SimilarityCalculator()
        
        query_embedding = np.array([1.0, 0.0])
        prompt_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        prompt_names = ["prompt_a"]  # Mismatch: 2 embeddings, 1 name
        
        with pytest.raises(ValueError, match="Number of prompt names.*doesn't match"):
            calc.find_top_k_similar(query_embedding, prompt_embeddings, prompt_names, top_k=1)
        
        # Invalid top_k
        prompt_names = ["prompt_a", "prompt_b"]
        with pytest.raises(ValueError, match="top_k must be positive"):
            calc.find_top_k_similar(query_embedding, prompt_embeddings, prompt_names, top_k=0)
        
        # Unknown method
        with pytest.raises(ValueError, match="Unknown similarity method"):
            calc.find_top_k_similar(query_embedding, prompt_embeddings, prompt_names, top_k=1, method="unknown")
    
    @pytest.mark.unit
    def test_filter_by_threshold_cosine(self):
        """Test filtering by threshold for cosine similarity."""
        calc = SimilarityCalculator(similarity_threshold=0.5)
        
        similarities = [
            ("prompt_a", 0.8),
            ("prompt_b", 0.3),
            ("prompt_c", 0.6),
        ]
        
        filtered = calc.filter_by_threshold(similarities, method="cosine")
        
        assert len(filtered) == 2
        assert ("prompt_a", 0.8) in filtered
        assert ("prompt_c", 0.6) in filtered
        assert ("prompt_b", 0.3) not in filtered
    
    @pytest.mark.unit
    def test_filter_by_threshold_euclidean(self):
        """Test filtering by threshold for Euclidean distance."""
        calc = SimilarityCalculator(similarity_threshold=2.0)
        
        similarities = [
            ("prompt_a", 1.0),  # Below threshold (good)
            ("prompt_b", 3.0),  # Above threshold (bad)
            ("prompt_c", 1.5),  # Below threshold (good)
        ]
        
        filtered = calc.filter_by_threshold(similarities, method="euclidean")
        
        assert len(filtered) == 2
        assert ("prompt_a", 1.0) in filtered
        assert ("prompt_c", 1.5) in filtered
        assert ("prompt_b", 3.0) not in filtered
    
    @pytest.mark.unit
    def test_filter_by_custom_threshold(self):
        """Test filtering with custom threshold."""
        calc = SimilarityCalculator(similarity_threshold=0.5)
        
        similarities = [("prompt_a", 0.7), ("prompt_b", 0.6)]
        
        # Use custom threshold higher than instance threshold
        filtered = calc.filter_by_threshold(similarities, threshold=0.65, method="cosine")
        
        assert len(filtered) == 1
        assert ("prompt_a", 0.7) in filtered
    
    @pytest.mark.unit
    def test_get_similarity_stats(self):
        """Test getting similarity statistics."""
        calc = SimilarityCalculator()
        
        query_embedding = np.array([1.0, 0.0, 0.0])
        prompt_embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ])
        
        stats = calc.get_similarity_stats(query_embedding, prompt_embeddings, method="cosine")
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert stats["method"] == "cosine"
        assert stats["num_prompts"] == 3
        
        # Check that max is close to 1.0 (perfect match)
        assert abs(stats["max"] - 1.0) < 1e-6
    
    @pytest.mark.unit
    def test_compare_methods(self):
        """Test comparing different similarity methods."""
        calc = SimilarityCalculator()
        
        query_embedding = np.array([1.0, 0.0, 0.0])
        prompt_embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        prompt_names = ["prompt_a", "prompt_b"]
        
        results = calc.compare_methods(query_embedding, prompt_embeddings, prompt_names, top_k=2)
        
        assert "cosine" in results
        assert "dot_product" in results
        assert "euclidean" in results
        
        # Each method should return 2 results
        assert len(results["cosine"]) == 2
        assert len(results["dot_product"]) == 2
        assert len(results["euclidean"]) == 2
        
        # For cosine and dot product, prompt_a should be first (higher similarity)
        assert results["cosine"][0][0] == "prompt_a"
        assert results["dot_product"][0][0] == "prompt_a"
        
        # For euclidean, prompt_a should be first (lower distance)
        assert results["euclidean"][0][0] == "prompt_a"

