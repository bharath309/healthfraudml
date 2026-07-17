"""HealthFraudML - Open-Source Healthcare Fraud Detection Framework."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="healthfraudml",
    version="0.1.0",
    author="Bharath Kumar Bahudhoddi",
    author_email="bharath.p90@gmail.com",
    description="A comprehensive ML framework for detecting financial fraud in healthcare claims",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bharath309/healthfraudml",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0,<3",
        "pandas>=1.3.0,<3",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "deep": ["tensorflow>=2.8.0", "torch>=1.10.0"],
        "explain": ["shap>=0.40.0", "lime>=0.2.0"],
        "rag": ["chromadb>=0.4.0", "google-genai>=0.1.0", "pypdf>=3.0.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=3.0.0", "flake8>=4.0.0"],
    },
)
