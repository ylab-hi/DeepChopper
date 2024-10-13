# Contributing to DeepChopper

We welcome contributions to DeepChopper! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:

   ```bash
   git clone https://github.com/ylab-hi/DeepChopper.git
   cd DeepChopper
   ```

3. Set up your development environment:

   ```bash
   conda env create -n environment.yaml
   conda activate deepchopper
   pip install pipx
   pipx install --suffix @master git+https://github.com/python-poetry/poetry.git@master
   poetry@master install
   ```

## Making Changes

1. Create a new branch for your feature or bugfix:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them with a clear commit message.

3. Push your changes to your fork on GitHub:

   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a pull request against the main repository.

## Coding Standards

- Follow PEP 8 guidelines for Python code.
- Write clear, concise comments and docstrings.
- Ensure your code is compatible with Python 3.10+.

## Testing

- Add unit tests for new functionality.
- Ensure all tests pass before submitting a pull request:

## Documentation

- Update the README.md if you're adding or changing features.
- Add or update docstrings for new or modified functions and classes.

## Submitting Pull Requests

1. Ensure your PR description clearly describes the problem and solution.
2. Include the relevant issue number if applicable.
3. Make sure all tests pass and the code lints without errors.

## Reporting Issues

- Use the GitHub issue tracker to report bugs or suggest features.
- Provide as much detail as possible, including steps to reproduce for bugs.

Thank you for contributing to DeepChopper!
