# Contributing to Sat3DGen

Thanks for your interest in **Sat3DGen**! 🎉

We warmly welcome contributions from anyone — researchers, engineers, and
hobbyists alike. Whether it is a bug fix, a feature, a documentation
improvement, or a benchmark/dataset extension, your help is appreciated.

## How to Contribute

### 1. Reporting Bugs

If you find a bug, please open an issue with:

- A clear description of the problem.
- A minimal reproducible example (command, input, expected vs actual output).
- Your environment (OS, Python version, PyTorch / CUDA version, GPU model).

### 2. Suggesting Enhancements

Open an issue tagged `enhancement` and describe:

- The motivation / use-case.
- A rough proposal of the API or behavior change.

### 3. Pull Requests

We accept pull requests for any of the following:

- Bug fixes.
- New features (please open an issue first to discuss large changes).
- Documentation improvements (`docs/`, `README.md`, code comments).
- Refactors that improve readability without changing behavior.
- Additional examples or tutorials.

**Workflow:**

1. Fork the repository and create a topic branch:
   ```bash
   git checkout -b feature/my-improvement
   ```
2. Make your changes with clear commit messages.
3. Make sure your code:
   - Runs without syntax errors (`python -m py_compile <file>`).
   - Follows the existing code style.
   - Includes English comments / docstrings (this project uses English-only).
4. Open a pull request describing what was changed and why.

### 4. Becoming a Maintainer

If you are interested in helping maintain the project (reviewing PRs,
triaging issues, releasing checkpoints, etc.), please reach out by opening
an issue titled `[maintainer] introduction`. We are happy to onboard new
maintainers from the community.

## Code Style

- **Language**: All comments, docstrings, log messages, and variable names
  must be in English.
- **Docstrings**: Use plain triple-quoted strings; brief one-liners are fine
  for small helpers, full descriptions for public APIs.
- **Comments**: Explain *why*, not *what*, when the code is non-obvious.
- **Formatting**: Follow PEP 8 where reasonable. We do not enforce a strict
  formatter, but please keep changes consistent with surrounding code.

## Acknowledgement

By submitting a pull request, you agree that your contribution will be
licensed under the project's [MIT License](LICENSE).

Thanks again for helping make Sat3DGen better! 💜
