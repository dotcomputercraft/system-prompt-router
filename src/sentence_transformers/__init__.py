"""Mock sentence_transformers for testing."""
import numpy as np

class SentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            # Return a single embedding
            return np.random.rand(384)
        else:
            # Return multiple embeddings
            return np.random.rand(len(texts), 384)
    
    def get_sentence_embedding_dimension(self):
        return 384
