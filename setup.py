from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="llm-blocks",
    version="0.3.4",
    author="Jamie Voynow",
    author_email="voynow99@gmail.com",
    description="Simple interface for creating and managing LLM chains",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/voynow/llm-blocks",
    packages=find_packages(),
    install_requires=[
        'langchain',
        'python-dotenv',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

# python setup.py sdist bdist_wheel
# twine upload dist/*