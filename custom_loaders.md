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