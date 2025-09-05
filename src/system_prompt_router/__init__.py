"""
System Prompt Router - A Python library for intelligent prompt routing using embeddings.

This package provides functionality to automatically route user queries to the most
appropriate system prompt using semantic similarity matching with sentence-transformers.
"""

from .router import SystemPromptRouter
from .library import PromptLibrary
from .embeddings import EmbeddingEngine
from .similarity import SimilarityCalculator
from .config import Config

__version__ = "1.0.0"
__author__ = "System Prompt Router Team"
__email__ = "contact@example.com"

__all__ = [
    "SystemPromptRouter",
    "PromptLibrary", 
    "EmbeddingEngine",
    "SimilarityCalculator",
    "Config",
]

