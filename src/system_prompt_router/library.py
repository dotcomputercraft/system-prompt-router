"""
Prompt library management for System Prompt Router.

This module handles loading, storing, and managing the collection of system prompts
used for routing user queries.
"""

import json
import yaml
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


class PromptLibrary:
    """Manages a collection of system prompts for routing."""
    
    def __init__(self):
        """Initialize empty prompt library."""
        self.prompts: Dict[str, Dict[str, str]] = {}
        self._descriptions: List[str] = []
        self._names: List[str] = []
    
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
        if not name or not name.strip():
            raise ValueError("Prompt name cannot be empty")
        
        if not description or not description.strip():
            raise ValueError("Prompt description cannot be empty")
        
        if not system_prompt or not system_prompt.strip():
            raise ValueError("System prompt cannot be empty")
        
        # Check for duplicate names
        if name in self.prompts:
            raise ValueError(f"Prompt with name '{name}' already exists")
        
        prompt_data = {
            "description": description.strip(),
            "system_prompt": system_prompt.strip(),
        }
        
        if metadata:
            prompt_data["metadata"] = metadata
        
        self.prompts[name] = prompt_data
        self._update_lists()
    
    def remove_prompt(self, name: str) -> bool:
        """
        Remove a prompt from the library.
        
        Args:
            name: Name of the prompt to remove
            
        Returns:
            True if prompt was removed, False if not found
        """
        if name in self.prompts:
            del self.prompts[name]
            self._update_lists()
            return True
        return False
    
    def get_prompt(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific prompt by name.
        
        Args:
            name: Name of the prompt
            
        Returns:
            Prompt data dictionary or None if not found
        """
        return self.prompts.get(name)
    
    def get_system_prompt(self, name: str) -> Optional[str]:
        """
        Get the system prompt text for a specific prompt.
        
        Args:
            name: Name of the prompt
            
        Returns:
            System prompt text or None if not found
        """
        prompt_data = self.get_prompt(name)
        return prompt_data["system_prompt"] if prompt_data else None
    
    def get_description(self, name: str) -> Optional[str]:
        """
        Get the description for a specific prompt.
        
        Args:
            name: Name of the prompt
            
        Returns:
            Description text or None if not found
        """
        prompt_data = self.get_prompt(name)
        return prompt_data["description"] if prompt_data else None
    
    def list_prompts(self) -> List[str]:
        """
        Get list of all prompt names.
        
        Returns:
            List of prompt names
        """
        return list(self.prompts.keys())
    
    def get_all_descriptions(self) -> List[str]:
        """
        Get all prompt descriptions in order.
        
        Returns:
            List of descriptions
        """
        return self._descriptions.copy()
    
    def get_all_names(self) -> List[str]:
        """
        Get all prompt names in order.
        
        Returns:
            List of prompt names
        """
        return self._names.copy()
    
    def load_from_dict(self, prompts_dict: Dict[str, Dict[str, Any]]) -> None:
        """
        Load prompts from a dictionary.
        
        Args:
            prompts_dict: Dictionary of prompts
        """
        for name, data in prompts_dict.items():
            if not isinstance(data, dict):
                raise ValueError(f"Prompt data for '{name}' must be a dictionary")
            
            if "description" not in data:
                raise ValueError(f"Prompt '{name}' missing 'description' field")
            
            if "system_prompt" not in data:
                raise ValueError(f"Prompt '{name}' missing 'system_prompt' field")
            
            metadata = data.get("metadata")
            self.add_prompt(name, data["description"], data["system_prompt"], metadata)
    
    def load_from_json(self, file_path: str) -> None:
        """
        Load prompts from a JSON file.
        
        Args:
            file_path: Path to JSON file
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.load_from_dict(data)
    
    def _update_lists(self) -> None:
        """Update internal lists of names and descriptions."""
        self._names = list(self.prompts.keys())
        self._descriptions = [self.prompts[name]["description"] for name in self._names]
    
    def clear(self) -> None:
        """Clear all prompts from the library."""
        self.prompts.clear()
        self._update_lists()
    
    def __len__(self) -> int:
        """Return number of prompts in library."""
        return len(self.prompts)
    
    def __contains__(self, name: str) -> bool:
        """Check if prompt name exists in library."""
        return name in self.prompts


    
    def get_library_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the prompt library.
        
        Returns:
            Dictionary with library statistics
        """
        if not self.prompts:
            return {
                "total_prompts": 0,
                "avg_description_length": 0,
                "avg_system_prompt_length": 0,
                "longest_description": "",
                "longest_system_prompt": "",
                "prompt_names": []
            }
        
        descriptions = [prompt["description"] for prompt in self.prompts.values()]
        system_prompts = [prompt["system_prompt"] for prompt in self.prompts.values()]
        
        return {
            "total_prompts": len(self.prompts),
            "avg_description_length": sum(len(d) for d in descriptions) / len(descriptions),
            "avg_system_prompt_length": sum(len(s) for s in system_prompts) / len(system_prompts),
            "longest_description": max(descriptions, key=len),
            "longest_system_prompt": max(system_prompts, key=len),
            "prompt_names": list(self.prompts.keys())
        }
    
    def validate_library(self) -> List[str]:
        """
        Validate the prompt library for common issues.
        
        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []
        
        if not self.prompts:
            errors.append("Library is empty - no prompts loaded")
            return errors
        
        for name, prompt in self.prompts.items():
            # Check for required fields
            if not prompt.get("description"):
                errors.append(f"Prompt '{name}' has empty description")
            
            if not prompt.get("system_prompt"):
                errors.append(f"Prompt '{name}' has empty system prompt")
            
            # Check for reasonable lengths
            description = prompt.get("description", "")
            if len(description) < 10:
                errors.append(f"Prompt '{name}' has very short description ({len(description)} chars)")
            
            system_prompt = prompt.get("system_prompt", "")
            if len(system_prompt) < 20:
                errors.append(f"Prompt '{name}' has very short system prompt ({len(system_prompt)} chars)")
        
        return errors

