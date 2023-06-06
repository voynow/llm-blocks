from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-blocks",
    version="0.1.4",
    author="Jamie Voynow",
    author_email="voynow99@gmail.com",
    description="Using LLMS to facilitate document retrieval and chat operations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/voynow/llm-blocks",
    packages=find_packages(),
    install_requires=[
        'langchain',
        'pandas',
        'matplotlib',
        'numpy',
        'ipywidgets',
        'uuid',
        'python-dotenv',
        'git2vec'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
