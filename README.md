# Custom Git Loaders in Langchain

This package provides two custom Git loaders, `FastGitLoader` and `TurboGitLoader`, that are more efficient than the default `GitLoader` in Langchain. These loaders can be found inside `custom_loaders.py`.

## FastGitLoader

The `FastGitLoader` is a slight modification of the default Langchain Git loader, with improvements in file filtering to speed up the loading process. Usage is similar to the default Git loader:

```python
from custom_loaders import FastGitLoader

loader = FastGitLoader(
    clone_url="https://github.com/hwchase17/langchain",
    repo_path="./example_data/FastGitLoader/",
    branch="master",
    file_filter=lambda file_path: file_path.endswith(".py"),
)

data = loader.load()
print(f"Length of data from dataloader: {len(data)}")  # takes about 2 minutes 10 seconds
```

## TurboGitLoader

The `TurboGitLoader` is a complete rewrite of the Langchain Git loader, taking advantage of parallelism to speed up the loading process even more:

```python
from custom_loaders import TurboGitLoader

loader = TurboGitLoader(
    clone_url="https://github.com/hwchase17/langchain",
    repo_path="./example_data/TurboGitLoader/",
    branch="master",
    file_filter=lambda file_path: file_path.endswith(".py"),
)

data = loader.load()
print(f"Length of data from dataloader: {len(data)}")  # takes about 20 seconds
```

### Performance Comparison

- FastGitLoader: ~30% improvement over Langchain's default Git loader
- TurboGitLoader: almost 90% improvement over Langchain's default Git loader

## Dataloader Test

The `dataloader_test.ipynb` notebook demonstrates how to use the custom Git loaders and run some tests with Langchain.

## Pinecone Test

The `pinecone_test.ipynb` notebook demonstrates how to use Pinecone to set up a vector search index and run tests with Langchain.

## Requirements

To run the code in this package, you need to install the following dependencies:

- PyGithub
- langchain
- pinecone-client
- pandas
- matplotlib
- numpy