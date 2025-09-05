"""
Embedding engine for System Prompt Router.

This module handles all embedding operations including text encoding,
caching, and model management using sentence-transformers.
"""

import os
import pickle
import hashlib
import numpy as np
from typing import List, Union, Optional, Dict, Any
from pathlib import Path
from sentence_transformers import SentenceTransformer

from .config import Config


class EmbeddingEngine:
    """Handles embedding generation and caching for text inputs."""
    
    def __init__(self, config: Config):
        """
        Initialize embedding engine.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model: Optional[SentenceTransformer] = None
        self._cache: Dict[str, np.ndarray] = {}
        
        # Set up cache directory
        if self.config.cache_embeddings:
            self.cache_dir = Path(self.config.cache_dir) / "embeddings"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()
    
    def _get_model(self) -> SentenceTransformer:
        """
        Get or load the sentence transformer model.
        
        Returns:
            Loaded SentenceTransformer model
        """
        if self.model is None:
            print(f"Loading embedding model: {self.config.embedding_model}")
            self.model = SentenceTransformer(self.config.embedding_model)
        return self.model
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for text.
        
        Args:
            text: Input text
            
        Returns:
            Cache key string
        """
        # Include model name in cache key to avoid conflicts
        cache_input = f"{self.config.embedding_model}:{text}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _load_cache(self) -> None:
        """Load embeddings cache from disk."""
        cache_file = self.cache_dir / "embeddings.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self._cache = pickle.load(f)
                print(f"Loaded {len(self._cache)} cached embeddings")
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
                self._cache = {}
    
    def _save_cache(self) -> None:
        """Save embeddings cache to disk."""
        if not self.config.cache_embeddings:
            return
            
        cache_file = self.cache_dir / "embeddings.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self._cache, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode single text into embedding vector.
        
        Args:
            text: Input text to encode
            
        Returns:
            Embedding vector as numpy array
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        if self.config.cache_embeddings and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Generate embedding
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        
        # Cache the result
        if self.config.cache_embeddings:
            self._cache[cache_key] = embedding
            # Save cache periodically (every 10 new embeddings)
            if len(self._cache) % 10 == 0:
                self._save_cache()
        
        return embedding
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple texts into embedding vectors.
        
        Args:
            texts: List of input texts to encode
            
        Returns:
            Array of embedding vectors
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty")
            
            cache_key = self._get_cache_key(text)
            if self.config.cache_embeddings and cache_key in self._cache:
                embeddings.append(self._cache[cache_key])
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            model = self._get_model()
            new_embeddings = model.encode(uncached_texts, convert_to_numpy=True)
            
            # Fill in the placeholders and cache new embeddings
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
                
                if self.config.cache_embeddings:
                    cache_key = self._get_cache_key(texts[idx])
                    self._cache[cache_key] = embedding
        
        # Save cache if we added new embeddings
        if uncached_texts and self.config.cache_embeddings:
            self._save_cache()
        
        return np.array(embeddings)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the model.
        
        Returns:
            Embedding dimension
        """
        model = self._get_model()
        return model.get_sentence_embedding_dimension()
    
    def clear_cache(self) -> None:
        """Clear the embeddings cache."""
        self._cache.clear()
        cache_file = self.cache_dir / "embeddings.pkl"
        if cache_file.exists():
            cache_file.unlink()
        print("Embeddings cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_size = len(self._cache)
        cache_file = self.cache_dir / "embeddings.pkl" if self.config.cache_embeddings else None
        cache_file_size = cache_file.stat().st_size if cache_file and cache_file.exists() else 0
        
        return {
            "cached_embeddings": cache_size,
            "cache_file_size_mb": cache_file_size / (1024 * 1024),
            "cache_enabled": self.config.cache_embeddings,
            "model_name": self.config.embedding_model,
            "embedding_dimension": self.get_embedding_dimension() if self.model else None,
        }
    
    def __del__(self):
        """Cleanup: save cache when object is destroyed."""
        if hasattr(self, '_cache') and self.config.cache_embeddings:
            self._save_cache()

