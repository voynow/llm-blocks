# Custom Git Loaders

This repository provides custom Git Loaders, which can speed up loading files from a Git repository compared to the default langchain GitLoader.

## Installation

To use these custom Git Loaders, you'll need to install the following Python packages if they're not already in your environment:

```
pip install GitPython
pip install langchain
```

## Loaders

There are two custom Git Loaders available:

- `FastGitLoader`: This is a slightly modified version of the default langchain GitLoader. It can load files faster by filtering out unwanted files earlier in the loading process.
- `TurboGitLoader`: This is a complete rewrite of the langchain GitLoader that uses parallelism to load files significantly faster.

## Usage

You can find examples of using the custom Git Loaders in the [dataloader_test.ipynb](dataloader_test.ipynb) notebook.

### FastGitLoader Example

```python
from custom_loaders import FastGitLoader

loader = FastGitLoader(
    clone_url="https://github.com/hwchase17/langchain",
    repo_path="./example_data/FastGitLoader/",
    branch="master",
    file_filter=lambda file_path: file_path.endswith(".py")
)
data = loader.load()
print(f"Length of data from dataloader: {len(data)}")
```

### TurboGitLoader Example

```python
from custom_loaders import TurboGitLoader

loader = TurboGitLoader(
    clone_url="https://github.com/hwchase17/langchain",
    repo_path="./example_data/TurboGitLoader/",
    branch="master",
    file_filter=lambda file_path: file_path.endswith(".py")
)
data = loader.load()
print(f"Length of data from dataloader: {len(data)}")
```

## Requirements

The complete list of required packages can be found in the [requirements.txt](requirements.txt) file.