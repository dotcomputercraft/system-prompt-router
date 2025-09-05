"""
Tests for the prompt library module.
"""

import json
import pytest
from pathlib import Path

from system_prompt_router.library import PromptLibrary


class TestPromptLibrary:
    """Test cases for PromptLibrary class."""
    
    @pytest.mark.unit
    def test_empty_library_initialization(self):
        """Test initialization of empty library."""
        library = PromptLibrary()
        
        assert len(library) == 0
        assert library.get_prompt_count() == 0
        assert library.list_prompts() == []
        assert library.get_all_descriptions() == []
        assert library.get_all_names() == []
    
    @pytest.mark.unit
    def test_add_prompt(self):
        """Test adding a single prompt."""
        library = PromptLibrary()
        
        library.add_prompt(
            name="test_prompt",
            description="A test prompt",
            system_prompt="You are a test assistant."
        )
        
        assert len(library) == 1
        assert "test_prompt" in library
        assert library.get_prompt_count() == 1
        
        prompt = library.get_prompt("test_prompt")
        assert prompt is not None
        assert prompt["description"] == "A test prompt"
        assert prompt["system_prompt"] == "You are a test assistant."
    
    @pytest.mark.unit
    def test_add_prompt_with_metadata(self):
        """Test adding a prompt with metadata."""
        library = PromptLibrary()
        metadata = {"category": "test", "difficulty": "easy"}
        
        library.add_prompt(
            name="test_prompt",
            description="A test prompt",
            system_prompt="You are a test assistant.",
            metadata=metadata
        )
        
        prompt = library.get_prompt("test_prompt")
        assert prompt["metadata"] == metadata
    
    @pytest.mark.unit
    def test_add_duplicate_prompt_raises_error(self):
        """Test that adding duplicate prompt raises error."""
        library = PromptLibrary()
        
        library.add_prompt("test", "desc", "prompt")
        
        with pytest.raises(ValueError, match="already exists"):
            library.add_prompt("test", "desc2", "prompt2")
    
    @pytest.mark.unit
    def test_add_prompt_validation(self):
        """Test prompt validation during addition."""
        library = PromptLibrary()
        
        # Empty name
        with pytest.raises(ValueError, match="name cannot be empty"):
            library.add_prompt("", "desc", "prompt")
        
        # Empty description
        with pytest.raises(ValueError, match="description cannot be empty"):
            library.add_prompt("name", "", "prompt")
        
        # Empty system prompt
        with pytest.raises(ValueError, match="system prompt cannot be empty"):
            library.add_prompt("name", "desc", "")
    
    @pytest.mark.unit
    def test_remove_prompt(self):
        """Test removing a prompt."""
        library = PromptLibrary()
        library.add_prompt("test", "desc", "prompt")
        
        assert len(library) == 1
        
        # Remove existing prompt
        result = library.remove_prompt("test")
        assert result is True
        assert len(library) == 0
        assert "test" not in library
        
        # Remove non-existent prompt
        result = library.remove_prompt("nonexistent")
        assert result is False
    
    @pytest.mark.unit
    def test_get_methods(self, prompt_library):
        """Test various get methods."""
        # Test get_system_prompt
        system_prompt = prompt_library.get_system_prompt("test_prompt_1")
        assert system_prompt == "You are a helpful coding assistant."
        
        # Test get_description
        description = prompt_library.get_description("test_prompt_1")
        assert description == "A test prompt for coding assistance"
        
        # Test get_prompt for non-existent
        assert prompt_library.get_prompt("nonexistent") is None
        assert prompt_library.get_system_prompt("nonexistent") is None
        assert prompt_library.get_description("nonexistent") is None
    
    @pytest.mark.unit
    def test_list_and_get_all_methods(self, prompt_library):
        """Test list and get_all methods."""
        names = prompt_library.list_prompts()
        assert len(names) == 3
        assert "test_prompt_1" in names
        assert "test_prompt_2" in names
        assert "test_prompt_3" in names
        
        all_names = prompt_library.get_all_names()
        assert all_names == names
        
        descriptions = prompt_library.get_all_descriptions()
        assert len(descriptions) == 3
        assert "A test prompt for coding assistance" in descriptions
    
    @pytest.mark.unit
    def test_load_from_dict(self, sample_prompts):
        """Test loading prompts from dictionary."""
        library = PromptLibrary()
        library.load_from_dict(sample_prompts)
        
        assert len(library) == 3
        assert "test_prompt_1" in library
        assert library.get_description("test_prompt_1") == "A test prompt for coding assistance"
    
    @pytest.mark.unit
    def test_load_from_dict_validation(self):
        """Test validation when loading from dictionary."""
        library = PromptLibrary()
        
        # Invalid data type
        with pytest.raises(ValueError, match="must be a dictionary"):
            library.load_from_dict({"test": "not_a_dict"})
        
        # Missing description
        with pytest.raises(ValueError, match="missing 'description' field"):
            library.load_from_dict({"test": {"system_prompt": "prompt"}})
        
        # Missing system_prompt
        with pytest.raises(ValueError, match="missing 'system_prompt' field"):
            library.load_from_dict({"test": {"description": "desc"}})
    
    @pytest.mark.unit
    def test_load_from_json(self, test_prompts_file):
        """Test loading prompts from JSON file."""
        library = PromptLibrary()
        library.load_from_json(str(test_prompts_file))
        
        assert len(library) == 3
        assert "test_prompt_1" in library
    
    @pytest.mark.unit
    def test_load_from_nonexistent_file(self):
        """Test loading from non-existent file raises error."""
        library = PromptLibrary()
        
        with pytest.raises(FileNotFoundError):
            library.load_from_json("nonexistent.json")
    
    @pytest.mark.unit
    def test_save_to_json(self, prompt_library, temp_dir):
        """Test saving prompts to JSON file."""
        output_file = temp_dir / "output.json"
        prompt_library.save_to_json(str(output_file))
        
        assert output_file.exists()
        
        # Load and verify
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        assert len(data) == 3
        assert "test_prompt_1" in data
        assert data["test_prompt_1"]["description"] == "A test prompt for coding assistance"
    
    @pytest.mark.unit
    def test_search_prompts(self, prompt_library):
        """Test searching prompts."""
        # Search by name
        results = prompt_library.search_prompts("prompt_1")
        assert len(results) == 1
        assert results[0][0] == "test_prompt_1"
        
        # Search by description
        results = prompt_library.search_prompts("coding")
        assert len(results) == 1
        assert results[0][0] == "test_prompt_1"
        
        # Search with no matches
        results = prompt_library.search_prompts("nonexistent")
        assert len(results) == 0
        
        # Case insensitive search
        results = prompt_library.search_prompts("CODING")
        assert len(results) == 1
    
    @pytest.mark.unit
    def test_clear(self, prompt_library):
        """Test clearing all prompts."""
        assert len(prompt_library) == 3
        
        prompt_library.clear()
        
        assert len(prompt_library) == 0
        assert prompt_library.list_prompts() == []
    
    @pytest.mark.unit
    def test_contains_and_len(self, prompt_library):
        """Test __contains__ and __len__ methods."""
        assert len(prompt_library) == 3
        assert "test_prompt_1" in prompt_library
        assert "nonexistent" not in prompt_library
    
    @pytest.mark.unit
    def test_iteration(self, prompt_library):
        """Test iteration over prompt names."""
        names = list(prompt_library)
        assert len(names) == 3
        assert "test_prompt_1" in names
        assert "test_prompt_2" in names
        assert "test_prompt_3" in names
    
    @pytest.mark.unit
    def test_get_library_stats(self, prompt_library):
        """Test getting library statistics."""
        stats = prompt_library.get_library_stats()
        
        assert stats["total_prompts"] == 3
        assert "avg_description_length" in stats
        assert "avg_system_prompt_length" in stats
        assert "longest_description" in stats
        assert "longest_system_prompt" in stats
        assert "prompt_names" in stats
        assert len(stats["prompt_names"]) == 3
    
    @pytest.mark.unit
    def test_get_library_stats_empty(self):
        """Test getting statistics for empty library."""
        library = PromptLibrary()
        stats = library.get_library_stats()
        
        assert stats["total_prompts"] == 0
        assert stats["avg_description_length"] == 0
        assert stats["avg_system_prompt_length"] == 0

