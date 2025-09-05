"""
Setup script for System Prompt Router package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="system-prompt-router",
    version="1.0.0",
    author="System Prompt Router Team",
    author_email="contact@example.com",
    description="Intelligent prompt routing using embeddings and semantic similarity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/system-prompt-router",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "cli": [
            "rich>=12.0.0",
            "tqdm>=4.64.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "system-prompt-router=system_prompt_router.cli:cli",
            "spr=system_prompt_router.cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "system_prompt_router": ["../config/*.json", "../config/*.yaml"],
    },
    keywords="ai, embeddings, prompt-routing, nlp, openai, sentence-transformers",
    project_urls={
        "Bug Reports": "https://github.com/example/system-prompt-router/issues",
        "Source": "https://github.com/example/system-prompt-router",
        "Documentation": "https://github.com/example/system-prompt-router/docs",
    },
)

