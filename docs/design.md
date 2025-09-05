# System Prompt Router - Project Design

## Overview
This project implements a System Prompt Routing application using Python and sentence-transformers embeddings. The application automatically routes user queries to the most appropriate system prompt using semantic similarity matching.

## Architecture

### Core Components

#### 1. SystemPromptRouter (Core Engine)
- **Location**: `src/system_prompt_router/router.py`
- **Purpose**: Main orchestrator for the routing logic
- **Key Methods**:
  - `add_prompt(name, description, system_prompt)`: Add individual prompts
  - `load_prompt_library(prompts)`: Load multiple prompts
  - `find_best_prompt(query, top_k=3)`: Find best matching prompts
  - `generate_response(query)`: Generate AI response using best prompt
  - `list_prompts()`: List all available prompts

#### 2. PromptLibrary (Data Management)
- **Location**: `src/system_prompt_router/library.py`
- **Purpose**: Manage prompt storage and retrieval
- **Features**:
  - Pre-built prompt collection
  - Custom prompt addition
  - Prompt validation
  - Export/import functionality

#### 3. EmbeddingEngine (Semantic Processing)
- **Location**: `src/system_prompt_router/embeddings.py`
- **Purpose**: Handle all embedding operations
- **Features**:
  - Query embedding generation
  - Prompt description embeddings
  - Caching for performance optimization
  - Multiple model support

#### 4. SimilarityCalculator (Matching Logic)
- **Location**: `src/system_prompt_router/similarity.py`
- **Purpose**: Calculate semantic similarity between queries and prompts
- **Features**:
  - Cosine similarity computation
  - Top-K selection
  - Ranking and scoring

#### 5. CLI Interface (User Interaction)
- **Location**: `src/system_prompt_router/cli.py`
- **Purpose**: Command-line interface for user interaction
- **Features**:
  - Interactive mode
  - Batch processing
  - Configuration management
  - Help and documentation

#### 6. Configuration Manager
- **Location**: `src/system_prompt_router/config.py`
- **Purpose**: Handle application configuration
- **Features**:
  - Environment variable management
  - Model selection
  - API key handling
  - Default settings

### Project Structure
```
system-prompt-router/
├── src/
│   └── system_prompt_router/
│       ├── __init__.py
│       ├── router.py              # Main router class
│       ├── library.py             # Prompt library management
│       ├── embeddings.py          # Embedding operations
│       ├── similarity.py          # Similarity calculations
│       ├── cli.py                 # Command-line interface
│       ├── config.py              # Configuration management
│       └── utils.py               # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_router.py             # Router tests
│   ├── test_library.py            # Library tests
│   ├── test_embeddings.py         # Embedding tests
│   ├── test_similarity.py         # Similarity tests
│   ├── test_cli.py                # CLI tests
│   ├── test_config.py             # Configuration tests
│   └── fixtures/                  # Test data and fixtures
├── config/
│   ├── default_prompts.json       # Default prompt library
│   ├── config.yaml                # Application configuration
│   └── .env.example               # Environment variables example
├── diagrams/
│   ├── architecture.puml          # PlantUML architecture diagram
│   └── architecture.png           # Rendered diagram
├── docs/
│   ├── design.md                  # This design document
│   ├── api_reference.md           # API documentation
│   └── user_guide.md              # User guide
├── examples/
│   ├── basic_usage.py             # Basic usage examples
│   ├── custom_prompts.py          # Custom prompt examples
│   └── batch_processing.py        # Batch processing examples
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── pyproject.toml                 # Modern Python packaging
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules
```

## Data Flow

1. **User Input**: User provides a query through CLI
2. **Configuration Loading**: System loads configuration and API keys
3. **Prompt Library Loading**: Available prompts are loaded from storage
4. **Query Embedding**: User query is converted to embedding vector
5. **Similarity Calculation**: Query embedding is compared with prompt embeddings
6. **Best Match Selection**: Top-K most similar prompts are identified
7. **Response Generation**: Best matching prompt is used with OpenAI API
8. **Result Delivery**: Generated response is returned to user

## Key Features

### 1. Semantic Matching
- Uses sentence-transformers for high-quality embeddings
- Cosine similarity for accurate matching
- Configurable similarity thresholds

### 2. Extensible Prompt Library
- Pre-built prompts for common use cases
- Easy addition of custom prompts
- JSON-based storage format
- Validation and error handling

### 3. Performance Optimization
- Embedding caching to avoid recomputation
- Efficient numpy operations
- Lazy loading of models
- Configurable batch processing

### 4. Comprehensive Testing
- Unit tests for all components
- Integration tests for end-to-end workflows
- Performance benchmarks
- Test fixtures and mock data

### 5. Configuration Management
- Environment-based configuration
- Multiple embedding model support
- OpenAI model selection
- Logging and debugging options

## Dependencies

### Core Dependencies
- `sentence-transformers>=2.2.0`: For embedding generation
- `openai>=1.0.0`: For AI response generation
- `numpy>=1.21.0`: For numerical operations
- `python-dotenv>=1.0.0`: For environment variable management

### Development Dependencies
- `pytest>=7.0.0`: Testing framework
- `pytest-cov>=4.0.0`: Coverage reporting
- `black>=22.0.0`: Code formatting
- `flake8>=5.0.0`: Code linting
- `mypy>=1.0.0`: Type checking

### Optional Dependencies
- `pyyaml>=6.0`: For YAML configuration files
- `click>=8.0.0`: For enhanced CLI functionality
- `rich>=12.0.0`: For beautiful terminal output

## Implementation Phases

### Phase 1: Core Functionality
- Implement SystemPromptRouter class
- Basic embedding and similarity operations
- Simple prompt library management

### Phase 2: CLI Interface
- Command-line argument parsing
- Interactive mode implementation
- Configuration file support

### Phase 3: Testing Framework
- Comprehensive unit tests
- Integration test suite
- Performance benchmarks

### Phase 4: Documentation and Examples
- API documentation
- User guide
- Example scripts and tutorials

## Quality Assurance

### Code Quality
- Type hints throughout the codebase
- Comprehensive docstrings
- PEP 8 compliance
- Code coverage > 90%

### Testing Strategy
- Unit tests for individual components
- Integration tests for complete workflows
- Mock external API calls for reliable testing
- Performance regression tests

### Documentation
- Inline code documentation
- API reference documentation
- User guides and tutorials
- Architecture diagrams

## Future Enhancements

### Potential Features
- Web-based interface
- Prompt performance analytics
- Multi-language support
- Custom embedding models
- Prompt versioning and management
- Integration with other AI services

### Scalability Considerations
- Database backend for large prompt libraries
- Distributed embedding computation
- API service deployment
- Monitoring and logging infrastructure

