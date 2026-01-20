# Contributing

**Guidelines for contributing to DeepChopper development.**

We welcome contributions! This guide covers setting up your development environment, coding standards, and the contribution process.

## Getting Started

1. Fork the repository on GitHub.

2. Clone your fork locally:

   ```bash
   git clone https://github.com/ylab-hi/DeepChopper.git
   cd DeepChopper
   ```

3. Set up your development environment:

   ```bash
   # Install uv package manager
   pip install uv
   
   # Install dependencies
   uv sync
   
   # Build and install in development mode
   maturin develop --release
   
   # Run tests to verify
   uv run pytest tests -k "not slow"
   ```

   Alternatively, with conda:

   ```bash
   # Create conda environment
   conda create -n deepchopper python=3.10
   conda activate deepchopper
   
   # Install uv and dependencies
   pip install uv
   uv sync
   
   # Build and install
   maturin develop --release
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
