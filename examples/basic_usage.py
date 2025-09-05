"""
Basic usage example for System Prompt Router.

This script demonstrates the basic workflow of:
1. Initializing the router
2. Loading the default prompt library
3. Processing a user query
4. Finding the best matching prompt
5. Generating a response
"""

import os
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from system_prompt_router.router import SystemPromptRouter


def main():
    """Main function to demonstrate basic usage."""
    print("--- Basic Usage Example ---")
    
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        return
    
    try:
        # 1. Initialize the router
        print("\n1. Initializing SystemPromptRouter...")
        router = SystemPromptRouter()
        
        # 2. Load the default prompt library
        print("\n2. Loading default prompt library...")
        default_prompts_path = Path(__file__).parent.parent / "config" / "default_prompts.json"
        router.load_prompt_library(str(default_prompts_path))
        print(f"Loaded {len(router.library)} prompts.")
        
        # 3. Define a user query
        query = "Write a Python function to sort a list of numbers."
        print(f"\n3. User Query: [32m{query}[0m")
        
        # 4. Find the best matching prompt
        print("\n4. Finding best matching prompts...")
        matches = router.find_best_prompt(query, top_k=3)
        
        if not matches:
            print("No matching prompts found.")
            return
        
        print("Top matches:")
        for name, score, _ in matches:
            print(f"  - Prompt: [36m{name}[0m, Score: [33m{score:.4f}[0m")
        
        # 5. Generate a response
        print("\n5. Generating response...")
        response_data = router.generate_response(query)
        
        print("\n--- Response --- ")
        print(f"[32m{response_data["response"]}[0m")
        print("\n--- Metadata ---")
        print(f"Matched Prompt: [36m{response_data["matched_prompt"]}[0m")
        print(f"Similarity Score: [33m{response_data["similarity_score"]:.4f}[0m")
        print(f"Model Used: {response_data["model_used"]}")
        print(f"Tokens Used: {response_data["tokens_used"]}")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()


