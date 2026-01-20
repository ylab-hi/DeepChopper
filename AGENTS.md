# Agent Guidelines for DeepChopper

This document provides essential information for AI coding agents working on the DeepChopper codebase.

## Project Overview

DeepChopper is a **hybrid Rust/Python project** - a genomic language model for chimera artifact detection in Nanopore direct RNA sequencing. The project uses:

- **Rust** (core library) - Fast bioinformatics processing, encoding, and file I/O
- **Python** (ML/AI layer) - PyTorch Lightning models, training, inference, CLI
- **PyO3/Maturin** - Rust-Python bindings for seamless integration

**Key Directories:**

- `src/` - Rust source code (lib + binaries)
- `deepchopper/` - Python source code (models, CLI, utils)
- `tests/` - Python and Rust tests
- `configs/` - Hydra configuration files for training

## Development Setup

```bash
# Install dependencies (using uv package manager)
uv sync

# Build Rust extensions and install in development mode
maturin develop --release

# Or use Makefile
make install
```

## Build Commands

### Rust

```bash
# Build library
cargo build
cargo build --release

# Build with maturin (creates Python wheels)
maturin build
maturin build --release
maturin develop --release  # Install into current virtualenv
```

### Python

```bash
# Install with development dependencies
uv sync

# Run the CLI
uv run deepchopper --help
deepchopper --help  # If installed
```

## Test Commands

### Run All Tests

```bash
# Makefile convenience (runs both Rust and Python tests)
make test

# Rust tests only
cargo nextest run
# Or fallback: cargo test

# Python tests only (skip slow tests)
uv run pytest tests -k "not slow"

# All Python tests including slow ones
uv run pytest tests
```

### Run Single Test

```bash
# Python - run specific test file
uv run pytest tests/test_rust.py

# Python - run specific test function
uv run pytest tests/test_rust.py::test_encode_fqs_to_parquet

# Python - run tests matching a pattern
uv run pytest tests -k "encode"

# Rust - run specific test
cargo nextest run test_name
cargo test test_name
```

### Test Markers

Python tests use pytest markers:

- `@pytest.mark.slow` - Slow tests (skipped by default)
- `@pytest.mark.smoke` - Quick smoke tests
- `@pytest.mark.failing` - Known failing tests

## Lint/Format Commands

### Python (Ruff)

```bash
# Format code
ruff format .
ruff format deepchopper/

# Lint and auto-fix
ruff check --fix .

# Check only (no fixes)
ruff check .
```

### Rust

```bash
# Format code
cargo fmt

# Run clippy linter
cargo clippy
cargo clippy -- -D warnings  # Treat warnings as errors
```

### Pre-commit Hooks

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Install hooks (runs automatically on git commit)
pre-commit install
```

## Code Style Guidelines

### Python

#### Imports

- Standard library first, then third-party, then local imports
- Use `from __future__ import annotations` for forward references
- Ruff automatically organizes imports (follows isort conventions)

```python
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import typer
from rich import print

from .utils import pylogger
```

#### Formatting

- **Line length:** 120 characters
- **Quote style:** Double quotes for strings
- **Indentation:** 4 spaces (no tabs)
- **Docstring style:** Google format

```python
def predict(
    data_path: Path,
    gpus: int = 0,
    batch_size: int = 12,
) -> None:
    """Predict chimera artifacts in sequencing data.

    Args:
        data_path: Path to input dataset (FASTQ or Parquet)
        gpus: Number of GPUs to use for inference
        batch_size: Batch size for processing

    Returns:
        None. Results are written to output directory.

    Raises:
        FileNotFoundError: If data_path does not exist
    """
    pass
```

#### Type Annotations

- **Required** for function signatures
- Use modern syntax: `list[str]` not `List[str]`, `dict[str, int]` not `Dict[str, int]`
- Use `TYPE_CHECKING` guard for expensive imports only needed for typing

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightning.pytorch import LightningDataModule

def setup_data(config: dict[str, str]) -> LightningDataModule:
    pass
```

#### Naming Conventions

- **Functions/variables:** `snake_case`
- **Classes:** `PascalCase`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private members:** `_leading_underscore`

#### Error Handling

- Use specific exceptions, not bare `except:`
- Provide helpful error messages with context

```python
from pathlib import Path
import typer

if not data_path.exists():
    typer.secho(f"❌ Error: Data path '{data_path}' does not exist.", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)
```

### Rust

#### Imports

- Group: std → external crates → local crates
- Use `use` statements, avoid glob imports except for preludes

```rust
use std::path::{Path, PathBuf};
use std::fs::File;

use anyhow::Result;
use rayon::prelude::*;

use crate::fq_encode::Encoder;
use crate::utils;
```

#### Formatting

- Use `cargo fmt` (rustfmt with default settings)
- **Indentation:** 4 spaces
- **Line length:** 100 characters (rustfmt default)

#### Type Annotations

- Explicit types for public APIs
- Can omit for obvious local variables
- Use `impl Trait` for return types when appropriate

```rust
pub fn encode_fastq(path: &Path, k: usize) -> Result<Vec<Record>> {
    let file = File::open(path)?;
    // Type inference for reader is fine
    let reader = BufReader::new(file);
    Ok(records)
}
```

#### Naming Conventions

- **Functions/variables:** `snake_case`
- **Types/Traits:** `PascalCase`
- **Constants:** `UPPER_SNAKE_CASE`
- **Lifetimes:** `'short` lowercase

#### Error Handling

- Use `anyhow::Result<T>` for functions that can fail
- Use `thiserror` for custom error types
- Propagate errors with `?` operator

```rust
use anyhow::{Result, Context};

pub fn read_config(path: &Path) -> Result<Config> {
    let contents = std::fs::read_to_string(path)
        .context(format!("Failed to read config from {:?}", path))?;
    
    let config: Config = serde_json::from_str(&contents)
        .context("Failed to parse config JSON")?;
    
    Ok(config)
}
```

#### Important: Use ahash for HashMap/HashSet

**CRITICAL:** The project uses `ahash` for better performance. `std::collections::HashMap` and `HashSet` are **disallowed** by clippy.

```rust
// ❌ WRONG - Will fail clippy
use std::collections::HashMap;
let map: HashMap<String, i32> = HashMap::new();

// ✅ CORRECT - Use ahash
use ahash::HashMap;
let map: HashMap<String, i32> = HashMap::default();
```

## Common Patterns

### PyO3 Bindings

Functions exposed to Python use PyO3 annotations:

```rust
#[pyfunction]
fn encode_qual(qual: String, qual_offset: u8) -> Vec<usize> {
    qual.as_bytes()
        .par_iter()
        .map(|&q| (q - qual_offset) as usize)
        .collect()
}

#[pyclass(name = "RecordData")]
struct PyRecordData(RecordData);

#[pymethods]
impl PyRecordData {
    #[new]
    fn py_new(id: String, seq: String) -> Self {
        Self(RecordData { id, seq })
    }
}
```

### Parallel Processing

Use `rayon` for parallel iteration in Rust:

```rust
use rayon::prelude::*;

let results: Vec<_> = input_data
    .par_iter()
    .map(|item| process(item))
    .collect();
```

## Testing Guidelines

- **Test coverage:** Aim for meaningful coverage, not just 100%
- **Python docstrings:** 80% coverage required (interrogate)
- **Test data:** Located in `tests/data/`
- Use `tmp_path` fixture in pytest for temporary files
- Mock external dependencies appropriately

## Additional Notes

- **Commit hooks:** Pre-commit will run formatters and linters automatically
- **CI/CD:** GitHub Actions runs tests on PRs (see `.github/workflows/`)
- **Documentation:** Update docstrings when changing public APIs
- **Version:** Managed by `bump-my-version` tool (see `.bumpversion.toml`)

## Quick Reference

```bash
# Complete development workflow
uv sync                              # Install dependencies
maturin develop --release            # Build Rust + install
uv run pytest tests -k "not slow"    # Run tests
cargo fmt && ruff format .           # Format code
cargo clippy && ruff check .         # Lint code
```

**Questions?** Check `README.md`, `CONTRIBUTING.md`, or `documentation/tutorial.md` for more details.
