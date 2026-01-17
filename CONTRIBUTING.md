# Contributing to this project

Glad that you are considering contributing to **AxionBloch**!
Contributions of all kinds are welcome: bug reports, documentation improvements,
examples, and code contributions.

## Reporting bugs and requesting features

Please open a GitHub issue and include:

- A clear description of the problem or request
- Your OS, Python version, and compiler (if relevant)
- Steps to reproduce the issue
- Error messages or stack traces, if any

Minimal reproducible examples are greatly appreciated.

## Development workflow

1. Fork the repository on GitHub
2. Create a new branch from `main`
3. Make your changes
4. Run tests locally
5. Open a pull request (PR) to `main`

Please keep PRs focused on a single issue or feature.

## Code formatting and style

We use pre-commit to enforce formatting:

- Python: Black, Ruff
- C/C++: clang-format

Please install hooks before contributing:
```bash
pre-commit install
