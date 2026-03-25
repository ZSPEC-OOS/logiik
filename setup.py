from setuptools import setup, find_packages

setup(
    name="cognita-ai",
    version="1.0.0",
    description="AI training system with teacher-student architecture and local knowledge storage",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.4.0",
        "openai>=1.0.0",
        "anthropic>=0.3.0",
        "chromadb>=0.4.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "streamlit>=1.25.0",
        "plotly>=5.15.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "cognita-api=cognita.api.server:app",
        ]
    },
)
