# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### ⚠️ Breaking Changes

- **DEPRECATED: `deepchopper encode` command removed**
  - The `encode` subcommand for converting FASTQ to Parquet format is now deprecated and removed
  - **Direct FASTQ input**: All DeepChopper commands now accept FASTQ files directly
  - **No preprocessing required**: Eliminates the need for intermediate Parquet conversion
  - **Simpler workflow**: `deepchopper predict` now works directly with `.fastq`, `.fq`, `.fastq.gz`, or `.fq.gz` files
  - **Migration**: Users should remove any `deepchopper encode` steps from their pipelines
  - **Why deprecated**: Parquet encoding added unnecessary complexity and storage overhead without significant performance benefits for typical use cases

### 🚀 Features

- **Direct FASTQ Processing**: Complete pipeline overhaul to work directly with FASTQ files
  - Automatic format detection (plain text, gzip, bgzip)
  - Streaming processing for memory efficiency
  - No intermediate file conversion required
  - Faster time-to-results by eliminating encoding step

### 📚 Documentation

- **Website Redesign**: Complete documentation website overhaul with modern blue theme
  - Migrated from teal/cyan to professional blue color scheme (#2563eb)
  - Improved readability and WCAG AA compliance
  - Cleaner navigation with TOC on right sidebar (disabled `toc.integrate`)
  - Shortened page titles for better navigation UX
- **Version Management**: Implemented versioned documentation with mike
  - Deployed version selector for easy switching between releases
  - Currently maintaining `dev` (development) and `1.2.9` (stable) versions
  - Automatic `dev` docs deployment on every push to main
  - Version-specific docs for each tagged release
- **CI/CD Optimization**: Enhanced GitHub Actions workflows
  - Added path filters to skip Python builds on docs-only changes
  - Saves ~20-40 CI minutes per documentation update
  - Consolidated documentation deployment into single unified workflow
  - Integrated with mike for automated version deployment
- **Navigation Improvements**: 
  - Disabled `navigation.tabs` in favor of native Material for MkDocs version selector
  - Improved left sidebar navigation with sections and expansion
  - Better mobile responsiveness
- **Content Updates**:
  - Fixed broken internal links in tutorial
  - Updated contributing guide with uv setup instructions
  - Updated installation documentation for current dependencies

### 🐛 Bug Fixes

- **Critical**: Replace deprecated `HuggingFaceDataset.from_dict()` with `.select()` for datasets>=3.0.0 compatibility in `only_fq.py`
  - Fixes compatibility issues with HuggingFace datasets library v3.0.0+
  - Prevents `ValueError: Feature type 'List' not found` errors
  - Affects sample limiting functionality in train/val/test dataset preparation
- **CI/CD**: Fix working directory inconsistency in GitHub Actions test workflows
  - Replace `working-directory:` with explicit `cd` commands for clarity
  - Ensures venv creation and activation happen in correct directory
- **CLI**: Improve deprecated `encode` command error messaging
  - Simplified function signature (removed unused parameters)
  - Added clear, colored error message directing users to `deepchopper predict`
  - Exit with code 1 to indicate error
- **Core**: Move random seed initialization after path validation in `predict()`
  - Prevents unnecessary seed setting when path validation fails
  - Improves early exit behavior and error handling

### ⚡ Performance Improvements

- **CI/CD**: Add Rust/Cargo build artifact caching to all GitHub Actions workflows
  - Cache `~/.cargo/` and `target/` directories across workflow runs
  - Expected 30-40% faster build times on cache hits
  - Applied to linux, musllinux, windows, and macos jobs in both release workflows
- **Data Pipeline**: Add intelligent memory management for large FASTQ files
  - Automatically reduces parallel workers for files >1GB (from cpu_count to max 4)
  - Prevents out-of-memory errors on large dataset processing
  - Logs file size and adjusted worker count for transparency
- **Data Pipeline**: Add comprehensive FASTQ file validation
  - Validates record completeness (name, sequence, quality string)
  - Validates sequence/quality length matching
  - Validates target parsing from sequence IDs
  - Early detection of corrupted or malformed FASTQ files
  - Clear error messages with file path and record position

### 🔧 Build & Tooling

- **Migration**: Migrate from Poetry to uv for dependency management

  - ⚡ 10-100x faster dependency resolution and installation
  - 📦 Better lock file performance (uv.lock vs poetry.lock)
  - 🔧 Simplified development workflow with `uv sync`
  - ✅ Remove poetry-plugin-export dependency

- **CI/CD Optimization**: Integrate uv into GitHub Actions workflows

  - Add `astral-sh/setup-uv@v5` action with caching enabled
  - Replace `pip` with `uv pip` for faster package installation in tests
  - Replace `python -m venv` with `uv venv` for faster virtual environments
  - ~50% faster CI runs with uv caching
  - Applied to both release-python.yml and release-python-cli.yml

### ⚠️ Testing Changes

- **ARM Architecture Testing**: ARM platform testing (aarch64, armv7, etc.) temporarily disabled in CI/CD workflows
  - ARM wheel builds are still created and published
  - ARM platform users should verify functionality after installation
  - To be re-enabled in future release with uv-compatible testing infrastructure

### 📚 Documentation

- Update CONTRIBUTING.md with uv setup instructions
- Add alternative setup method without conda
- Update CHANGELOG.md with comprehensive list of bug fixes and improvements

## [py-cli-v1.2.9] - 2025-11-17

### 🐛 Bug Fixes

- **Critical**: Fix syntax error in CLI predict function ternary operator
- **Compatibility**: Replace deprecated `HuggingFaceDataset.from_dict()` with `.select()` for datasets>=3.0.0
- **Dependencies**: Resolve PyTorch 2.6.x compatibility by adding torchvision>=0.21.0
- **Type System**: Fix `Parameter.make_metavar()` error by pinning typer\<0.13.0 and click\<8.2.0
- **Warnings**: Replace deprecated pynvml with nvidia-ml-py>=12.0.0

### 🔧 Build & CI

- Optimize Rust release profile for broader CPU compatibility (Windows/macOS)
  - Change `lto = true` to `lto = "thin"` for faster builds
  - Increase `codegen-units` from 1 to 16 for better compatibility
  - Add `RUSTFLAGS` for x86-64 baseline compatibility on Windows
  - Add dynamic symbol lookup for macOS Python ABI compatibility

### 📦 Dependencies

- Add `torchvision>=0.21.0` (required for PyTorch 2.6.x)
- Pin `typer>=0.12.0,<0.13.0` (avoid breaking API changes)
- Pin `click>=8.1.0,<8.2.0` (ensure compatibility with typer)
- Add `nvidia-ml-py>=12.0.0` (replace deprecated pynvml)

### 💡 Migration Notes

**Breaking Changes**:

- ⚠️ **Data Format Incompatibility**: Parquet files generated with deepchopper v1.2.8 and earlier are **NOT compatible** with v1.2.9 due to datasets library schema changes
- The legacy `'List'` schema has been removed in favor of `'Sequence'` schema (datasets>=3.0.0 requirement)

**Action Required**:

- **CRITICAL**: You must regenerate all parquet data files before using v1.2.9
- Users experiencing "illegal instruction" errors on Windows should update to this version
- Users seeing "symbol not found" errors on macOS should update to this version
- Users with existing parquet files will encounter `ValueError: Feature type 'List' not found` errors

**Migration Steps**:

1. Update to v1.2.9: `pip install --upgrade deepchopper-cli`
2. **Regenerate all parquet files** from your original FASTQ data sources
3. Use the new parquet files for training/prediction workflows
4. Remove deprecated `pynvml` if manually installed: `pip uninstall -y pynvml`

**Why Regeneration is Required**:

- The HuggingFace `datasets` library v3.0.0+ removed support for the legacy `'List'` feature type
- Parquet files store schema metadata that cannot be automatically converted at runtime
- Attempting to load old parquet files will result in immediate schema validation errors

### 🔍 Technical Details

This release addresses several critical compatibility issues:

1. **datasets Library Compatibility**: The HuggingFace `datasets` library removed support for the legacy `'List'` feature type in version 3.0.0. The codebase now uses `.select()` instead of `.from_dict()` to avoid schema conversion issues.

2. **PyTorch Ecosystem**: PyTorch 2.6.x requires torchvision 0.21.x for proper operation. Missing this dependency caused runtime errors with CUDA/GPU operations.

3. **Typer/Click API Changes**: Typer 0.13+ introduced breaking changes to the `Parameter.make_metavar()` API. Pinning to compatible versions ensures CLI stability.

4. **Cross-Platform Binary Compatibility**: Optimized Rust compiler settings to generate binaries that work on a wider range of CPUs, particularly addressing issues on GitHub Actions runners.

## [py-cli-v1.2.8] - 2025-11-17

### 💼 Other

- Update the macos sys version
- Fix abi incompability in macos and windows
- Bump verison to v1.2.8

## [py-cli-v1.2.7] - 2025-11-17

### 🚀 Features

- Add poetry-plugin-export
- Add new model configuration for hyena experiment
- Seed everything
- Add TODO for memory optimization in prediction script
- Implement memory optimization and streaming processing in prediction script
- Enhance temporary file handling in prediction script
- Add memory usage tracking in prediction script
- Enhance DeepChopper class with model loading and pushing methods
- Enhance CLI options for prediction and update documentation

### 🐛 Bug Fixes

- Fix incorrect usage of references in function arguments
- Update pre-commit hooks versions
- Update macOS runner version to 13
- Update data paths and workers count
- Update dependencies versions and Rust toolchain channel
- Update versions in Cargo.toml
- Update log_cli value in pyproject.toml to boolean
- Change log level for sequence prediction truncation

### 💼 Other

- Update thiserror requirement from 1.0 to 2.0
- Update noodles requirement from 0.84.0 to 0.85.0
- *(deps)* Bump actions/attest-build-provenance from 1 to 2
- Update noodles requirement from 0.87.0 to 0.88.0
- Update noodles requirement from 0.88.0 to 0.90.0
- Update rand requirement from 0.8 to 0.9
- Update rand_distr requirement from 0.4 to 0.5
- Update noodles requirement from 0.90.0 to 0.91.0
- Update noodles requirement from 0.91.0 to 0.93.0
- Update dependencies and actions versions
- Update needletail requirement from 0.5 to 0.6
- Update pyo3-build-config requirement from 0.24 to 0.25
- *(deps)* Bump actions/download-artifact from 4 to 5
- *(deps)* Bump actions/checkout from 4 to 5
- *(deps)* Bump rayon from 1.10 to 1.11 and clap from 4.5.43 to 4.5.45
- *(deps)* Update noodles to version 0.101.0 and refactor fastq reader usage
- *(deps)* Bump actions/attest-build-provenance from 2 to 3
- *(deps)* Bump actions/setup-python from 5 to 6
- Update bio requirement from 2.3 to 3.0
- Update pyo3-build-config requirement from 0.25 to 0.27
- Upgrade candle
- *(deps)* Bump actions/download-artifact from 5 to 6
- *(deps)* Bump actions/upload-artifact from 4 to 5
- Bump deepchopper-cli version

### 🚜 Refactor

- Update dependencies versions and function signatures
- Update data type conversion for quality score in tokenizer

### 📚 Documentation

- Update troubleshooting section in tutorial
- Update tutorial.md with memory error solution
- Update tutorial with additional information
- Add parameter optimization guidelines to documentation
- Improve memory usage documentation in predict.rs

### ⚙️ Miscellaneous Tasks

- Add default configuration file and git-cliff template
- Update dependencies versions and script paths
- Upgrade dependencies
- Upgrade dependencies
- Add gitignore
- Rename pytest steps to Test CLI and refactor workflow for clarity
- Update workflow to improve clarity and remove redundant steps
- Update dependencies and improve documentation
- Update configuration files for model training
- Update sysinfo dependency version in Cargo.toml
- Remove outdated checkpoint file
- Update pre-commit configuration and enhance model upload guide

## [py-cli-v1.2.6] - 2024-11-05

### 🚀 Features

- Add function to split dataset with both adapters
- Allow to parse multi targets

### 🐛 Bug Fixes

- Update datasets and evaluate versions
- Fix typo
- Update usage of vectorize_targets

### 💼 Other

- *(deps)* Bump crazy-max/ghaction-github-labeler from 5.0.0 to 5.1.0
- Update noodles requirement from 0.83.0 to 0.84.0
- Update project versions to 1.2.6

### 🚜 Refactor

- Add multiprocessing context to improve performance
- Update data["target"] to use .numpy() for consistency

### 📚 Documentation

- Update documentation for DeepChopper model architecture
- Update citation link in README.md

### 🎨 Styling

- Update pre-commit configs and CLI help message
- Update table formatting in README.md
- Update meta tags and author information
- Fix typo in tutorial.md output flag typo
- Update authors in pyproject.toml and LICENSE
- Update CLI help messages for DeepChopper features

### ⚙️ Miscellaneous Tasks

- Update labels and workflows configurations

## [py-cli-v1.2.5] - 2024-10-18

### 🚀 Features

- Update version to 1.2.5

### 📚 Documentation

- Update version in **init**.py and bumpversion files
- Add compatibility matrices for Conda and PyPI installations
- Add link to PyPI support section
- Update compatibility matrices format
- Update installation documentation in README

### 🎨 Styling

- Update typer.Option syntax in CLI file

## [py-cli-v1.2.4beta] - 2024-10-15

### 🚀 Features

- Update version to 1.2.4

## [py-cli-v1.2.4] - 2024-10-14

### 🚀 Features

- Update dependencies versions for rayon, walkdir, pyo3, bstr, lazy_static, tempfile, parquet, arrow, flate2, and clap
- Add Samtools for BAM to FASTQ conversion
- Update documentation and repository URLs

### 📚 Documentation

- Add link to updated documentation in README.md
- Update troubleshooting section in tutorial
- Add CLI documentation

### 🎨 Styling

- Fix typos in tutorial.md commands
- Update Rust flags order and add release profile settings
- Simplify error message in FqDataModule
- Update tutorial section title for clarity

### 🧪 Testing

- Add raw data to test file

## [py-cli-v1.2.3] - 2024-10-13

### 🚀 Features

- Bump project version to 1.2.3 in Cargo.toml and pyproject.toml

### 🐛 Bug Fixes

- Update dependencies versions to latest versions
- Update package versions to 1.2.1

### 🚜 Refactor

- Update dependencies versions in pyproject.toml

### 🎨 Styling

- Remove unnecessary empty lines

## [py-cli-v1.2.2] - 2024-10-13

### 🚀 Features

- Update project versions to 1.2.2

### 🚜 Refactor

- Update CLI options in chop function
- Add type check for fastq_path and data_path

### 🎨 Styling

- Update variable name in predict function

## [py-cli-v1.2.1] - 2024-10-12

### 🐛 Bug Fixes

- Update version to 1.2.1 in multiple files

### 🚜 Refactor

- Update project versions to 1.2.0

### 📚 Documentation

- Update installation dependencies and tutorial content

### 🎨 Styling

- Remove commented out dependency

## [py-cli-v1.2.0] - 2024-10-11

### 🚀 Features

- Update current_version to "1.0.1" in bumpversion config

### 🚜 Refactor

- Update version numbers to 1.1.0 in configuration files

### 📚 Documentation

- Update Quick Start link in README.md
- Update tutorial and add logging function

### 🎨 Styling

- Fix indentation and variable naming convention
- Update Python versions in workflows and pyproject.toml

### ⚙️ Miscellaneous Tasks

- Update deepchopper-cli dependency to version 1.0.1
- Update Python version to 3.10 and build args
- Update Python versions for build jobs

## [py-cli-v1.0.1] - 2024-10-11

### 🐛 Bug Fixes

- Update project versions to 1.0.1

## [py-cli-v1.0.0] - 2024-10-11

### 🐛 Bug Fixes

- Update version to 1.0.0 in multiple files

## [py-cli-v0.1.0] - 2024-10-11

### 🐛 Bug Fixes

- Update Python version to 3.10 in CI workflow

### 📚 Documentation

- Add bumpversion configuration file

### ⚡ Performance

- Improve Python CLI release build performance

### 🎨 Styling

- Update PythonVersion to use quotes

### ⚙️ Miscellaneous Tasks

- Update PyPI release condition
- Remove redundant pytest job configurations

## [deepchopper-v0.1.0] - 2024-10-11

### 🚀 Features

- Add logo images for DeepChopper
- Update derive_builder version and add lexical dependency
- Add error module and EncodingError type
- Add functions to convert between k-mer target region and original target region
- *(output)* Add functions for generating and removing intervals
- Add split and writefq modules
- Add function to write FASTQ records
- *(output)* Add functions to write fastq records
- Add function to encode multiple FASTQ files
- Add function `summary_record_len` to Python module
- Add pynvim dependency
- Implement Display trait for FqEncoder and FqEncoderOption
- Add Python script for running deepchopper
- Add FqEncoder class and its dependencies
- *(src/fq_encode.rs)* Add logging statements
- Add 1000_records.fq.gz test data
- *(output)* Add function to extract records by IDs
- Add logging task and test_log() function
- Add .gitkeep file to notebooks folder
- Add support for parallel file processing
- Add functions to encode FASTQ files
- *(utils)* Add function to load kmer to ID mapping
- *(utils)* Add functions to load and save SafeTensors
- Add serde_json dependency and encode to JSON format
- Add ParquetData struct and refactor encode_fq_to_parquet function
- Add function to encode FQ path to Parquet
- Add encode_parqut task for Parquet encoding
- Add function to encode FASTQ paths to JSON
- Add custom build function in TensorEncoderBuilder
- Add datasets and ipykernel dependencies
- Add seqeval dependency
- Add test_input.parquet file
- Add tokenizers module
- Add tokenizers to pyproject.toml and update load_dataset in tasks.py
- Add data split functionality to test function
- Add KmerOverlapPreTokenizer for tokenization
- Add KmerPreTokenizer and KmerDecoder classes
- Add torchinfo to pyproject.toml
- Add evaluate package "^0.4.1"
- Add training functionality for deepchopper model
- Add hyena model and quality usage to training file
- *(utils)* Add function to collect and split dataset
- Add hyena_model_train to .gitignore and update dataset loading function
- Add dataset splitting function and test cases
- Update PyTorch versions and add NVIDIA CUDA packages
- Update training script with new dataset and options
- Update dependencies and training configurations
- Add functions for label region extraction and smoothing
- Update training script for 600000 samples and 15 epochs
- Add import statement for typing Annotated
- Add show_metrics option to load_trainer function
- Add optional output_path parameter to main function
- Add chunking functionality for Parquet encoding
- Add logging and encoding features
- Add optimizations for release build configurations
- Create folder for chunk parquet files
- Add functions to load model and dataset from checkpoint
- Add web UI functionality and launch interface
- Add prediction function for DNA sequence processing
- Add new checkpoint files and configurations
- Improve performance by using parallel iterator for mapping
- Add function to collect and split dataset with prefix
- Add BenchmarkCNN model and LitBenchmarkCNN class
- Add configuration files for FqDataModule and CNN model
- Add max_train_samples, max_val_samples, max_test_samples
- Update model_checkpoint filename pattern
- Update model hyperparameters and configurations
- *(configs)* Update early stopping patience and batch sizes
- Update scheduler type and add transformer model
- Add TokenClassificationModel with transformer encoder
- Add configurations for Hyena and Transformer models
- Update transformer model configuration
- Add CustomWriter callback for writing predictions
- Update data path and number of workers in eval script
- Add smooth module and majority voting function
- Add candle-core dependency and new smooth utils
- Add test_predicts function in smooth.rs
- *(python, smooth/predict, smooth/stat)* Add function and struct properties
- Add parallel reading of BAM records
- Update code for collecting statistics and overlap results
- Save more data to StatResult
- Add max_process_intervals to OverlapOptions
- Add ContinuousIntervalLoss criterion to models
- Add option for specifying number of threads
- Add diff, blat and chop
- Update dependencies and default values
- Add KanModel class for deep learning model
- Update dependencies and refactor function signatures in DeepChopper
- Add new functionality for reverse complement function
- Add new features for selecting reads by type and name
- Update sample name and data folder paths
- Add new predict script and adjust data settings
- Remove parse name @region|id
- Add new script fq2fa.rs
- Add new prediction task for HEK293T_RNA004 dataset
- Add option to filter records by length
- Update noodles dependency to version 0.78.0
- Add Precision-Recall Curve metric to TokenClassificationLit model
- Introduce new function to read records from file
- Update sample_name and data_folder variables
- Add script for computing distance matrix
- Add option for selecting chop type
- Add script chop option ct, mcr, and new binary replace
- Update candle-core version to 0.6.0, add HashSet usage
- Update project name and description
- Add script for generating fusion circle plot
- Update pre-commit repos to latest versions
- Add chop command to DeepChopper CLI
- Add DeepChopper model class for pretrained models
- Add UI functionality and improve predict CLI
- Add file input option for dataset loading
- Update DeepChopper commands and documentation
- Add main function to run CLI app
- Update tutorial with GPU support for predicting reads
- Update dependencies in pyproject.toml
- Add new dependencies for project
- Add scikit-learn dependency
- Add train and eval modules, update dependencies
- Add maturin configuration to pyproject.toml
- Add new CLI module for DeepChopper

### 🐛 Bug Fixes

- *(src/output/writefq.rs)* Add missing argument in function call
- Update maturin command in Makefile
- Handle FileNotFoundError with logging instead of raising
- Update pytorch-cu121 URL to cu118
- Update ruff-pre-commit to v0.2.2
- Fix invalid index assignment in vertorize_target function
- Update forward method signature in CNN model
- Update noodles crate version to 0.68.0
- Update batch size to 32
- Update batch size to 18
- Update data_path and num_workers in eval script
- Update condition to check for intervals_number \<= 4
- Remove read with unmap, supp, secondary
- Correct indexing in id_list2seq functions
- Use pyo3 new features
- Correct typos in function names and update dependencies in Cargo.toml
- Accepet gzip fq, bgzip fq and fq
- Update sample name in predict script to match model number
- Update precision and recall calculation methods
- Update predict CLI output path handling
- Update pre-commit hook versions
- Update CLI command to use new tool name
- Fix default value for number of GPUs
- Update default value for gpus to "auto"
- Update predict_data_path with as_posix() method
- Update gradio version to ^5.0.1
- Update method signature in DeepChopper class
- Update versions for noodles and candle-core
- Fix getting qname from record in read_bam_records
- Fix import path in encode_fq.py
- Update requires-python to support Python 3.10+

### 💼 Other

- Remove unnecessary crate-type "rlib"
- Add GitHub actions and labels configuration files

### 🚜 Refactor

- Update package name in **init**.py
- Remove unused code and improve code readability
- Update target region calculation and error message
- Move types into separate file
- Rename `KmerTable` to `Kmer2IdTable` and add `Id2KmerTable`
- Rename function `generate_records_by_remove_interval` to `split_records_by_remove_interval`
- *(fq_encode)* Remove unused variable and refactor code
- *(fq_encode)* Change type of 'qual' parameter to Matrix
- *(test)* Refactor test_rust.py
- Update function signatures and error handling
- Rename references to "panda" to "deepchopper"
- *(python)* Refactor kmerids_to_seq function
- *(option)* Refactor FqEncoderOption structfeat(run.py): add save to npz
- Clean and rename files, update tasks file
- Clean up and improve logging and code structure
- Improve error handling and data processing
- Remove debug and warn log statements
- Remove unnecessary code and update dependencies
- Modify FqEncoder to use noodles for FASTQ reading
- Update function names and data encoding in encode_fq.py
- Update variable names in FqEncoderOption structure
- Simplify crate-type definition and function names
- Remove unnecessary code and improve performance
- Consolidate crate-type and encode_qual functions
- Update encoder_seq method to support overlap option
- Remove commented-out code and optimize kmer processing
- Remove unused code and optimize vectorization
- Update logging in fq_encode/triat.rs and Cargo.toml
- Add imports and remove redundant code
- Remove unused imports and variables
- Remove commented-out code and unused imports
- Simplify dataset sampling and processing
- Remove unused kmer_size parameter in encode_fq.py
- Update import statement in predict.py
- Update dependencies and refactor code for efficiency
- Update function name and arguments in predict.py
- Improve unmaped intervals generation
- Update output directory variable to use Path object
- Update function parameters for smooth_label_region
- Remove unnecessary import in test module
- Update CNN model configurations and structure
- Update imports and metrics in CNN module
- Update file paths to use pathlib in FqDataModule
- Update configurations and model parameters
- Update accuracy calculation and loss computation in CNN module
- Update evaluation config and model in deepchopper
- Update model and module names, refactor class names
- Update CNN model architecture and forward method
- Update batch size and model filter configurations
- Rename train files to hg_train files
- Simplify error message handling in model files
- Update model import paths to llm
- Add prediction functionality and configurations
- *(utils)* Update function arguments to accept i64 types
- *(utils)* Replace summary_predict_i64 with summary_predict_generic
- Add merge method to StatResult impl
- Increase line length to 120 and update logging levels
- Remove unnecessary print statements and add input parameters
- Add support for SA tag in BAM records
- Update HashMap imports to use ahash crate
- Update fastq writer instantiation in utils.rs
- Update function name and improve code readability
- Update model and checkpoint paths for vcap_004
- Update record filtering logic to use parallel iterators
- Add error context for reading records and names
- Improve reading and filtering of records
- Improve name selection logic
- Improve code readability and efficiency
- Update function return type and variable names
- Remove unused code and comments
- Update deepchopper CLI predict function
- Update Cargo.toml and remove unnecessary imports
- Update function parameters in deepchopper/ui/main.py
- Improve dataset loading logic and input processing
- Improve data loading and processing in predict function
- Rename ui function to web
- Improve code formatting in README.md
- Improve handling of GPU availability and selection
- Remove unnecessary modules from package import
- Update converting byte slices to vectors
- Update import statement for cli in **init**.py

### 📚 Documentation

- Add documentation for token classification model
- Add function documentation for predict.py
- Add function documentation for highlighting targets
- Add module docstrings for TokenClassificationLit and BenchmarkCNN
- Add Sherlock logo images
- Update DeepChopper README and tutorial
- Update link to tutorial file in README
- Update documentation content in index.html
- Update tutorial.md with corrected URLs and commands
- Update README with GPU flag usage
- Update storage space recommendation in tutorial
- Update license to Apache License Version 2.0
- Add GitHub workflow for releasing Python CLI
- Add LICENSE and README.md files
- Update license and add readme file path

### 🎨 Styling

- Update README.md logo URL
- Remove unnecessary code and print statements
- Update dependencies and clean up code bases
- Remove unnecessary blank lines and print statement
- Update Cargo.toml and src/lib.rs formatting
- Add debug log statement in FqEncoder
- Update Cargo.toml crate-type formatting
- Update crate-type in Cargo.toml and line-length in pyproject.toml
- Improve code formatting for better readability
- Update pre-commit hook version to v0.3.0
- Update crate-type in Cargo.toml and clean up code
- Update training script configurations
- Improve command help messages and option annotations
- Update help message capitalization
- Improve code formatting and remove redundant print
- Update formatting for crate type and function arguments
- Update code formatting and remove unused imports
- Update monitor metrics to "val/f1" and remove unused imports
- Add example input array to TokenClassificationLit
- Remove unnecessary comment and empty lines
- Update crate-type formatting in Cargo.toml
- Update README.md formatting and text
- Update paths and sample names in scripts
- Update rustflags for target x86_64-apple-darwin
- Improve file creation comments and paths in code
- Remove unnecessary target.x86_64-apple-darwin config
- Remove unnecessary references in ParquetEncoder
- Fix formatting in basic_module.py
- Remove unused import from fq2fa.rs
- Add clippy lint allowances in several functions
- Improve ChopType display and comparison
- Update argument type and add annotation for clarity
- Reorder imports for better readability
- Remove unused imports and comments
- Improve code formatting and remove unnecessary lines
- Update versions for gradio and fastapi
- Improve command-line interface instructions
- Update variable type and add custom CSS styling
- Remove unnecessary file exclusion in .gitignore
- Update project description and license in pyproject.toml
- Update typer and gradio versions
- Comment out unused import statements

### 🧪 Testing

- *(fa)* Add tests for reading fasta and fastx files
- Add test for reading fastq file
- Add test for encoding large size FASTQ file
- Add test for encoding FASTQ files from paths
- Add test for encoding FASTQ to JSON conversion
- Rename summar_bam_record_len to summary_bam_record_len
- Tokenize and align labels and quals for training dataset

### ⚙️ Miscellaneous Tasks

- Add test_log function for logging
- Update .gitignore to exclude massif.out.\*
- Add task.slurm to .gitignore
- Update exclude paths in pyproject.toml
- Update .gitignore for wandb and checkpoints
- Update file paths for hg train scripts
- Remove unnecessary code block in notebook
- Update depen
- Add setup_panic to eval.rs and select.rs
- Update file paths in chop.sh and internal.rs files
- Remove unnecessary checkpoint files
- Remove unused test files
- Remove redundant Windows x86 target
- Remove unnecessary target configurations
- Remove redundant aarch64 platform configurations
- Remove unused musllinux job and related steps
- Update dependencies and remove unused binaries
- Update concurrency configuration and CLI import
- Update PyO3/maturin-action arguments for releasing Python CLI

<!-- generated by git-cliff -->
