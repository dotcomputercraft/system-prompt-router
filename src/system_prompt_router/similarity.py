"""
Similarity calculator for System Prompt Router.

This module handles similarity calculations between query embeddings and
prompt embeddings using cosine similarity and other metrics.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityCalculator:
    """Calculates similarity between embeddings and ranks results."""
    
    def __init__(self, similarity_threshold: float = 0.5):
        """
        Initialize similarity calculator.
        
        Args:
            similarity_threshold: Minimum similarity score to consider a match
        """
        self.similarity_threshold = similarity_threshold
    
    def calculate_cosine_similarity(
        self, 
        query_embedding: np.ndarray, 
        prompt_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity between query and prompt embeddings.
        
        Args:
            query_embedding: Query embedding vector (1D array)
            prompt_embeddings: Prompt embedding matrix (2D array)
            
        Returns:
            Array of similarity scores
        """
        if query_embedding.ndim != 1:
            raise ValueError("Query embedding must be 1-dimensional")
        
        if prompt_embeddings.ndim != 2:
            raise ValueError("Prompt embeddings must be 2-dimensional")
        
        if query_embedding.shape[0] != prompt_embeddings.shape[1]:
            raise ValueError(
                f"Embedding dimensions don't match: query={query_embedding.shape[0]}, "
                f"prompts={prompt_embeddings.shape[1]}"
            )
        
        # Reshape query embedding for sklearn
        query_reshaped = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_reshaped, prompt_embeddings)[0]
        
        return similarities
    
    def calculate_dot_product_similarity(
        self, 
        query_embedding: np.ndarray, 
        prompt_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate dot product similarity between query and prompt embeddings.
        
        Args:
            query_embedding: Query embedding vector (1D array)
            prompt_embeddings: Prompt embedding matrix (2D array)
            
        Returns:
            Array of similarity scores
        """
        if query_embedding.ndim != 1:
            raise ValueError("Query embedding must be 1-dimensional")
        
        if prompt_embeddings.ndim != 2:
            raise ValueError("Prompt embeddings must be 2-dimensional")
        
        # Calculate dot product
        similarities = np.dot(prompt_embeddings, query_embedding)
        
        return similarities
    
    def calculate_euclidean_distance(
        self, 
        query_embedding: np.ndarray, 
        prompt_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Euclidean distance between query and prompt embeddings.
        
        Args:
            query_embedding: Query embedding vector (1D array)
            prompt_embeddings: Prompt embedding matrix (2D array)
            
        Returns:
            Array of distance scores (lower is more similar)
        """
        if query_embedding.ndim != 1:
            raise ValueError("Query embedding must be 1-dimensional")
        
        if prompt_embeddings.ndim != 2:
            raise ValueError("Prompt embeddings must be 2-dimensional")
        
        # Calculate Euclidean distance
        distances = np.linalg.norm(prompt_embeddings - query_embedding, axis=1)
        
        return distances
    
    def find_top_k_similar(
        self,
        query_embedding: np.ndarray,
        prompt_embeddings: np.ndarray,
        prompt_names: List[str],
        top_k: int = 3,
        method: str = "cosine"
    ) -> List[Tuple[str, float]]:
        """
        Find top-k most similar prompts to the query.
        
        Args:
            query_embedding: Query embedding vector
            prompt_embeddings: Prompt embedding matrix
            prompt_names: List of prompt names corresponding to embeddings
            top_k: Number of top results to return
            method: Similarity method ("cosine", "dot_product", "euclidean")
            
        Returns:
            List of tuples (prompt_name, similarity_score) sorted by similarity
        """
        if len(prompt_names) != prompt_embeddings.shape[0]:
            raise ValueError(
                f"Number of prompt names ({len(prompt_names)}) doesn't match "
                f"number of embeddings ({prompt_embeddings.shape[0]})"
            )
        
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        # Calculate similarities based on method
        if method == "cosine":
            similarities = self.calculate_cosine_similarity(query_embedding, prompt_embeddings)
            # Higher is better for cosine similarity
            top_indices = np.argsort(similarities)[::-1][:top_k]
            scores = similarities[top_indices]
        elif method == "dot_product":
            similarities = self.calculate_dot_product_similarity(query_embedding, prompt_embeddings)
            # Higher is better for dot product
            top_indices = np.argsort(similarities)[::-1][:top_k]
            scores = similarities[top_indices]
        elif method == "euclidean":
            distances = self.calculate_euclidean_distance(query_embedding, prompt_embeddings)
            # Lower is better for Euclidean distance
            top_indices = np.argsort(distances)[:top_k]
            scores = distances[top_indices]
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        # Create results list
        results = []
        for idx, score in zip(top_indices, scores):
            prompt_name = prompt_names[idx]
            results.append((prompt_name, float(score)))
        
        return results
    
    def filter_by_threshold(
        self,
        similarities: List[Tuple[str, float]],
        threshold: float = None,
        method: str = "cosine"
    ) -> List[Tuple[str, float]]:
        """
        Filter similarity results by threshold.
        
        Args:
            similarities: List of (prompt_name, score) tuples
            threshold: Similarity threshold (uses instance threshold if None)
            method: Similarity method used ("cosine", "dot_product", "euclidean")
            
        Returns:
            Filtered list of similarities above threshold
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        filtered_results = []
        
        for prompt_name, score in similarities:
            if method == "euclidean":
                # For Euclidean distance, lower is better
                if score <= threshold:
                    filtered_results.append((prompt_name, score))
            else:
                # For cosine and dot product, higher is better
                if score >= threshold:
                    filtered_results.append((prompt_name, score))
        
        return filtered_results
    
    def get_similarity_stats(
        self,
        query_embedding: np.ndarray,
        prompt_embeddings: np.ndarray,
        method: str = "cosine"
    ) -> Dict[str, float]:
        """
        Get statistical information about similarities.
        
        Args:
            query_embedding: Query embedding vector
            prompt_embeddings: Prompt embedding matrix
            method: Similarity method to use
            
        Returns:
            Dictionary with similarity statistics
        """
        if method == "cosine":
            similarities = self.calculate_cosine_similarity(query_embedding, prompt_embeddings)
        elif method == "dot_product":
            similarities = self.calculate_dot_product_similarity(query_embedding, prompt_embeddings)
        elif method == "euclidean":
            similarities = self.calculate_euclidean_distance(query_embedding, prompt_embeddings)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        return {
            "mean": float(np.mean(similarities)),
            "std": float(np.std(similarities)),
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
            "median": float(np.median(similarities)),
            "method": method,
            "num_prompts": len(similarities),
        }
    
    def compare_methods(
        self,
        query_embedding: np.ndarray,
        prompt_embeddings: np.ndarray,
        prompt_names: List[str],
        top_k: int = 3
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compare different similarity methods for the same query.
        
        Args:
            query_embedding: Query embedding vector
            prompt_embeddings: Prompt embedding matrix
            prompt_names: List of prompt names
            top_k: Number of top results for each method
            
        Returns:
            Dictionary with results for each method
        """
        methods = ["cosine", "dot_product", "euclidean"]
        results = {}
        
        for method in methods:
            try:
                results[method] = self.find_top_k_similar(
                    query_embedding, prompt_embeddings, prompt_names, top_k, method
                )
            except Exception as e:
                results[method] = f"Error: {str(e)}"
        
        return results

