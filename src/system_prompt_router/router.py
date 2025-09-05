"""
Main System Prompt Router implementation.

This module contains the main SystemPromptRouter class that orchestrates
all components to provide intelligent prompt routing functionality.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from openai import OpenAI

from .config import Config
from .library import PromptLibrary
from .embeddings import EmbeddingEngine
from .similarity import SimilarityCalculator


class SystemPromptRouter:
    """
    Main router class that matches user queries to appropriate system prompts
    using semantic similarity and generates responses using OpenAI API.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        embedding_model: Optional[str] = None,
        openai_model: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the System Prompt Router.
        
        Args:
            config: Configuration object (creates default if None)
            embedding_model: Override embedding model name
            openai_model: Override OpenAI model name
            openai_api_key: Override OpenAI API key
        """
        # Initialize configuration
        if config is None:
            config = Config()
        self.config = config
        
        # Override config values if provided
        if embedding_model:
            self.config.set("embedding_model", embedding_model)
        if openai_model:
            self.config.set("openai_model", openai_model)
        if openai_api_key:
            self.config.set("openai_api_key", openai_api_key)
        
        # Initialize components
        self.library = PromptLibrary()
        self.embedding_engine = EmbeddingEngine(self.config)
        self.similarity_calculator = SimilarityCalculator(self.config.similarity_threshold)
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_api_base
        )
        
        # Cache for prompt embeddings
        self._prompt_embeddings: Optional[np.ndarray] = None
        self._embeddings_computed = False
    
    def add_prompt(
        self, 
        name: str, 
        description: str, 
        system_prompt: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a single prompt to the library.
        
        Args:
            name: Unique name for the prompt
            description: Description used for similarity matching
            system_prompt: The actual system prompt text
            metadata: Optional metadata for the prompt
        """
        self.library.add_prompt(name, description, system_prompt, metadata)
        # Reset embeddings cache since library changed
        self._embeddings_computed = False
        self._prompt_embeddings = None
    
    def load_prompt_library(self, prompts: Union[Dict[str, Dict[str, Any]], str]) -> None:
        """
        Load multiple prompts into the library.
        
        Args:
            prompts: Dictionary of prompts or path to JSON/YAML file
        """
        if isinstance(prompts, str):
            # Load from file
            if prompts.endswith('.json'):
                self.library.load_from_json(prompts)
            elif prompts.endswith(('.yaml', '.yml')):
                self.library.load_from_yaml(prompts)
            else:
                raise ValueError("File must be JSON or YAML format")
        else:
            # Load from dictionary
            self.library.load_from_dict(prompts)
        
        # Reset embeddings cache since library changed
        self._embeddings_computed = False
        self._prompt_embeddings = None
    
    def _compute_prompt_embeddings(self) -> None:
        """Compute embeddings for all prompt descriptions."""
        if self._embeddings_computed and self._prompt_embeddings is not None:
            return
        
        if len(self.library) == 0:
            raise ValueError("No prompts loaded. Please add prompts first.")
        
        descriptions = self.library.get_all_descriptions()
        print(f"Computing embeddings for {len(descriptions)} prompts...")
        
        self._prompt_embeddings = self.embedding_engine.encode_texts(descriptions)
        self._embeddings_computed = True
        
        print(f"Computed embeddings with shape: {self._prompt_embeddings.shape}")
    
    def find_best_prompt(
        self, 
        user_query: str, 
        top_k: int = None,
        method: str = "cosine",
        return_scores: bool = True
    ) -> List[Tuple[str, float, str]]:
        """
        Find the best matching prompts for a user query.
        
        Args:
            user_query: The user's input query
            top_k: Number of top matches to return (uses config default if None)
            method: Similarity method ("cosine", "dot_product", "euclidean")
            return_scores: Whether to include similarity scores
            
        Returns:
            List of tuples (prompt_name, similarity_score, system_prompt)
        """
        if not user_query or not user_query.strip():
            raise ValueError("User query cannot be empty")
        
        if len(self.library) == 0:
            raise ValueError("No prompts loaded. Please add prompts first.")
        
        if top_k is None:
            top_k = self.config.top_k_results
        
        # Compute prompt embeddings if needed
        self._compute_prompt_embeddings()
        
        # Generate query embedding
        query_embedding = self.embedding_engine.encode_text(user_query)
        
        # Find similar prompts
        prompt_names = self.library.get_all_names()
        similarities = self.similarity_calculator.find_top_k_similar(
            query_embedding, self._prompt_embeddings, prompt_names, top_k, method
        )
        
        # Build results with system prompts
        results = []
        for prompt_name, score in similarities:
            system_prompt = self.library.get_system_prompt(prompt_name)
            if return_scores:
                results.append((prompt_name, score, system_prompt))
            else:
                results.append((prompt_name, system_prompt))
        
        return results
    
    def generate_response(
        self,
        user_query: str,
        use_best_prompt: bool = True,
        custom_system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using the best matching prompt or a custom prompt.
        
        Args:
            user_query: The user's input query
            use_best_prompt: Whether to use the best matching prompt
            custom_system_prompt: Custom system prompt to use instead
            max_tokens: Maximum tokens for response (uses config default if None)
            temperature: Temperature for response generation (uses config default if None)
            
        Returns:
            Dictionary containing response and metadata
        """
        if not user_query or not user_query.strip():
            raise ValueError("User query cannot be empty")
        
        # Determine system prompt to use
        if custom_system_prompt:
            system_prompt = custom_system_prompt
            matched_prompt = "custom"
            similarity_score = None
        elif use_best_prompt:
            if len(self.library) == 0:
                raise ValueError("No prompts loaded. Please add prompts first.")
            
            best_matches = self.find_best_prompt(user_query, top_k=1)
            if not best_matches:
                raise ValueError("No matching prompts found")
            
            matched_prompt, similarity_score, system_prompt = best_matches[0]
        else:
            raise ValueError("Either use_best_prompt must be True or custom_system_prompt must be provided")
        
        # Set generation parameters
        if max_tokens is None:
            max_tokens = self.config.get("max_tokens", 1000)
        if temperature is None:
            temperature = self.config.get("temperature", 0.7)
        
        # Generate response using OpenAI
        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            generated_text = response.choices[0].message.content
            
            return {
                "response": generated_text,
                "matched_prompt": matched_prompt,
                "similarity_score": similarity_score,
                "system_prompt": system_prompt,
                "model_used": self.config.openai_model,
                "tokens_used": response.usage.total_tokens if response.usage else None,
                "user_query": user_query
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {str(e)}")
    
    def list_prompts(self) -> Dict[str, str]:
        """
        List all available prompts with their descriptions.
        
        Returns:
            Dictionary mapping prompt names to descriptions
        """
        return {name: self.library.get_description(name) for name in self.library.list_prompts()}
    
    def get_prompt_details(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific prompt.
        
        Args:
            name: Name of the prompt
            
        Returns:
            Dictionary with prompt details or None if not found
        """
        return self.library.get_prompt(name)
    
    def remove_prompt(self, name: str) -> bool:
        """
        Remove a prompt from the library.
        
        Args:
            name: Name of the prompt to remove
            
        Returns:
            True if prompt was removed, False if not found
        """
        removed = self.library.remove_prompt(name)
        if removed:
            # Reset embeddings cache since library changed
            self._embeddings_computed = False
            self._prompt_embeddings = None
        return removed
    
    def clear_prompts(self) -> None:
        """Clear all prompts from the library."""
        self.library.clear()
        self._embeddings_computed = False
        self._prompt_embeddings = None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the router and its components.
        
        Returns:
            Dictionary with various statistics
        """
        library_stats = self.library.get_library_stats() if len(self.library) > 0 else {}
        embedding_stats = self.embedding_engine.get_cache_stats()
        
        return {
            "library": library_stats,
            "embeddings": embedding_stats,
            "config": {
                "embedding_model": self.config.embedding_model,
                "openai_model": self.config.openai_model,
                "similarity_threshold": self.config.similarity_threshold,
                "top_k_results": self.config.top_k_results,
            },
            "cache_status": {
                "embeddings_computed": self._embeddings_computed,
                "prompt_embeddings_shape": self._prompt_embeddings.shape if self._prompt_embeddings is not None else None,
            }
        }
    
    def compare_similarity_methods(
        self, 
        user_query: str, 
        top_k: int = 3
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compare different similarity methods for a query.
        
        Args:
            user_query: The user's input query
            top_k: Number of top results for each method
            
        Returns:
            Dictionary with results for each similarity method
        """
        if not user_query or not user_query.strip():
            raise ValueError("User query cannot be empty")
        
        if len(self.library) == 0:
            raise ValueError("No prompts loaded. Please add prompts first.")
        
        # Compute prompt embeddings if needed
        self._compute_prompt_embeddings()
        
        # Generate query embedding
        query_embedding = self.embedding_engine.encode_text(user_query)
        
        # Compare methods
        prompt_names = self.library.get_all_names()
        return self.similarity_calculator.compare_methods(
            query_embedding, self._prompt_embeddings, prompt_names, top_k
        )
    
    def validate_setup(self) -> List[str]:
        """
        Validate the router setup and configuration.
        
        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []
        
        # Check OpenAI API key
        if not self.config.openai_api_key:
            errors.append("OpenAI API key is not configured")
        
        # Check if prompts are loaded
        if len(self.library) == 0:
            errors.append("No prompts loaded in the library")
        
        # Validate library
        library_errors = self.library.validate_library()
        errors.extend(library_errors)
        
        # Test embedding model
        try:
            test_embedding = self.embedding_engine.encode_text("test")
            if test_embedding is None or len(test_embedding) == 0:
                errors.append("Embedding model is not working properly")
        except Exception as e:
            errors.append(f"Embedding model error: {str(e)}")
        
        return errors

