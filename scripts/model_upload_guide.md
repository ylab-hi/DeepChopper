# DeepChopper Model Upload Guide

This guide explains how to use the enhanced `modle2hub.py` script to push DeepChopper models to Hugging Face Hub with comprehensive metadata.

## Features

### Enhanced Script (`modle2hub.py`)

The script now includes:

1. **Comprehensive Model Card Generation**

   - Automatic extraction of F1 scores from checkpoint filenames
   - Detailed model architecture documentation
   - Training details and hyperparameters
   - Usage examples and limitations
   - Proper citations and references
   - Contact information and support links
   - **Automatic upload of README.md to Hugging Face Hub**

2. **Two-Stage Upload Process**

   - Stage 1: Upload model weights using `push_to_hub()`
   - Stage 2: Upload model card (README.md) using Hugging Face Hub API
   - Ensures both model and documentation are properly synced

3. **Rich CLI Interface**

   - Beautiful terminal output with Rich formatting
   - Interactive confirmations
   - Progress indicators for long operations
   - Detailed error messages with troubleshooting tips

4. **Advanced Options**

   - Custom F1 score specification
   - Custom model descriptions
   - Private repository support
   - Custom commit messages
   - HF API token support

5. **Validation and Error Handling**

   - Checkpoint file validation
   - Clear error messages
   - Troubleshooting guidance
   - Automatic cleanup of temporary files

## Usage

### Basic Usage

```bash
python scripts/modle2hub.py epoch_012_f1_0.9947.ckpt
```

This will:

- Auto-detect the F1 score (0.9947) from the filename
- Use the default model name: `yangliz5/deepchopper`
- Generate a comprehensive model card
- Push to Hugging Face Hub (public repository)

### Advanced Usage

```bash
python scripts/modle2hub.py epoch_012_f1_0.9947.ckpt \
    --model-name username/deepchopper-v2 \
    --f1-score 0.9947 \
    --description "Custom model description" \
    --commit-message "Upload DeepChopper v2.0" \
    --private \
    --token YOUR_HF_TOKEN
```

### Options

- `--model-name, -m`: Model repository ID (format: username/model-name)
- `--f1-score, -f`: F1 score to include in the model card
- `--description, -d`: Custom model description
- `--commit-message, -c`: Git commit message
- `--private, -p`: Make the repository private
- `--token, -t`: Hugging Face API token (or use HF_TOKEN env var)

## Model Card Content

The generated model card includes:

### Metadata (YAML frontmatter)

- Tags: genomics, bioinformatics, nanopore, rna-sequencing, chimera-detection, etc.
- License: MIT
- Language: DNA
- Pipeline tag: token-classification
- Library name: deepchopper

### Sections

1. **Model Details**: Architecture, authors, paper references
2. **Model Architecture**: Detailed breakdown of layers and dimensions
3. **Uses**: Direct use cases and downstream applications
4. **Training Details**: Data, procedure, hyperparameters
5. **Evaluation**: Metrics and results
6. **How to Use**: Installation and usage examples (Python API, CLI, Web UI)
7. **Limitations**: Platform-specific constraints
8. **Citation**: BibTeX format
9. **Contact & Support**: Links to documentation and issues

## Enhanced DeepChopper Class

The `DeepChopper` class in `deepchopper/models/dc_hg.py` has been improved with:

1. **Comprehensive Documentation**

   - Detailed docstrings for all methods
   - Usage examples
   - Parameter descriptions
   - Return type documentation

2. **Enhanced `to_hub` Method**

   - Support for commit messages
   - Private repository option
   - Token authentication
   - Proper type hints

3. **Better Type Safety**

   - Modern Python type hints (using `|` for unions)
   - Keyword-only arguments for better API design

## Prerequisites

Before uploading, ensure you have:

1. **Hugging Face Account**

   ```bash
   huggingface-cli login
   ```

2. **Required Dependencies**

   ```bash
   pip install huggingface-hub rich typer
   ```

3. **Valid Checkpoint File**

   - Should be a PyTorch Lightning checkpoint (.ckpt)
   - Contains trained model weights

## How It Works

### Upload Process

The script performs a two-stage upload:

1. **Model Upload**

   - Loads the checkpoint using `DeepChopper.from_checkpoint()`
   - Pushes model weights and configuration to Hugging Face Hub
   - Uses PyTorch Lightning's serialization format

2. **Model Card Upload**

   - Generates comprehensive README.md with metadata
   - Uses Hugging Face Hub API to upload the model card
   - Creates a separate commit for documentation
   - Ensures proper rendering on the Hugging Face model page

This two-stage approach ensures that both the model and its documentation are properly synchronized on Hugging Face Hub.

## Example Workflow

1. Train your DeepChopper model
2. Identify the best checkpoint (e.g., `epoch_012_f1_0.9947.ckpt`)
3. Login to Hugging Face: `huggingface-cli login`
4. Run the upload script:
   ```bash
   python scripts/modle2hub.py epoch_012_f1_0.9947.ckpt \
       --model-name username/deepchopper-finetuned
   ```
5. Review the model card preview (optional)
6. Confirm the upload
7. Script uploads model weights (Stage 1)
8. Script uploads model card README.md (Stage 2)
9. Share your model at `https://huggingface.co/username/deepchopper-finetuned`

## Troubleshooting

### Authentication Errors

- Run `huggingface-cli login` and enter your token
- Or pass `--token YOUR_TOKEN` to the script
- Or set `HF_TOKEN` environment variable

### Checkpoint Loading Errors

- Ensure the checkpoint is a valid PyTorch Lightning checkpoint
- Check that the file is not corrupted
- Verify the model architecture matches

### PyTorch 2.6+ Compatibility

**Note:** PyTorch 2.6 changed the default value of `weights_only` in `torch.load` from `False` to `True` for security reasons. DeepChopper checkpoints contain optimizer/scheduler configurations (`functools.partial`) which require `weights_only=False` to load.

**This is already handled in the code** - the `from_checkpoint` method explicitly sets `weights_only=False`. Only load checkpoints from trusted sources.

If you encounter errors like:

```
WeightsUnpickler error: Unsupported global: GLOBAL functools.partial
```

This means you're using an older version of the code. Update to the latest version which includes the fix.

### Network Errors

- Check your internet connection
- Ensure Hugging Face Hub is accessible
- Try with a smaller test repository first

### Model Card Not Showing

If you uploaded a model but the model card (README.md) is not visible:

1. **Check the upload logs**: Look for "Uploading model card (README.md)..." in the progress output
2. **Verify on Hugging Face**: Go to your model page and check the "Files and versions" tab
3. **Manual fix**: You can manually upload the README.md:
   ```bash
   # The script creates a temporary README.md, you can regenerate it
   python scripts/modle2hub.py YOUR_CHECKPOINT.ckpt --model-name username/model-name
   # Then manually upload at https://huggingface.co/username/model-name/tree/main
   ```
4. **Re-run with latest version**: Make sure you're using the latest version of the script that includes the two-stage upload process

## Best Practices

1. **Naming Convention**

   - Use descriptive model names: `username/deepchopper-{variant}`
   - Include version numbers for multiple versions

2. **Documentation**

   - Add F1 scores for performance tracking
   - Include custom descriptions for special models
   - Update commit messages to describe changes

3. **Privacy**

   - Use `--private` for sensitive or unpublished models
   - Make public only when ready for community use

4. **Version Control**

   - Tag releases in Git before uploading
   - Keep track of which checkpoint corresponds to which HF model

## Additional Resources

- [DeepChopper Documentation](https://github.com/ylab-hi/DeepChopper/blob/main/documentation/tutorial.md)
- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub/index)
- [DeepChopper Paper](https://www.biorxiv.org/content/10.1101/2024.10.23.619929v2)
