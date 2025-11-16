"""Script to push DeepChopper models to Hugging Face Hub with comprehensive metadata."""

import os
from pathlib import Path
from typing import Optional

import deepchopper
import typer
from huggingface_hub import HfApi
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Push DeepChopper models to Hugging Face Hub with comprehensive metadata.",
)
console = Console()


def create_model_card(
    model_name: str,
    ckpt_path: str,
    f1_score: Optional[float] = None,
    description: Optional[str] = None,
) -> str:
    """Generate a comprehensive model card for the Hugging Face Hub.

    Args:
        model_name: The name/identifier for the model on Hugging Face Hub
        ckpt_path: Path to the checkpoint file
        f1_score: Optional F1 score from checkpoint filename or manual input
        description: Optional custom description

    Returns:
        A formatted markdown string containing the model card
    """
    # Try to extract F1 score from checkpoint filename if not provided
    if f1_score is None:
        ckpt_filename = Path(ckpt_path).name
        if "f1" in ckpt_filename.lower():
            try:
                # Extract number after 'f1' or 'f1_'
                import re
                match = re.search(r'f1[_-]?([\d.]+)', ckpt_filename, re.IGNORECASE)
                if match:
                    f1_score = float(match.group(1))
            except (ValueError, AttributeError):
                pass

    # Default description
    if description is None:
        description = (
            "DeepChopper is a genomic language model designed to accurately detect and remove "
            "chimera artifacts in Nanopore direct RNA sequencing data. It uses a HyenaDNA backbone "
            "with a token classification head to identify artificial adapter sequences within reads."
        )

    model_card = f"""---
tags:
- genomics
- bioinformatics
- nanopore
- rna-sequencing
- chimera-detection
- token-classification
- hyenadna
- pytorch
- lightning
license: apache-2.0
datasets:
- nanopore-drna-seq
language:
- dna
library_name: deepchopper
pipeline_tag: token-classification
---

# DeepChopper: Chimera Detection for Nanopore Direct RNA Sequencing

{description}

## Model Details

### Model Description

DeepChopper leverages the HyenaDNA-small-32k backbone, a genomic foundation model, combined with a specialized token classification head to detect chimeric artifacts in nanopore direct RNA sequencing reads. The model processes both sequence information and base quality scores to make accurate predictions.

- **Developed by:** YLab Team ([Li et al., 2024](https://www.biorxiv.org/content/10.1101/2024.10.23.619929v2))
- **Model type:** Token Classification
- **Language(s):** DNA sequences
- **License:** Apache 2.0
- **Base Model:** HyenaDNA-small-32k-seqlen
- **Repository:** [DeepChopper GitHub](https://github.com/ylab-hi/DeepChopper)
- **Paper:** [A Genomic Language Model for Chimera Artifact Detection](https://www.biorxiv.org/content/10.1101/2024.10.23.619929v2)

### Model Architecture

- **Backbone:** HyenaDNA-small-32k (256 dimensions)
- **Classification Head:**
  - Linear Layer 1: 256 → 1024 dimensions
  - Linear Layer 2: 1024 → 1024 dimensions
  - Output Layer: 1024 → 2 classes (artifact/non-artifact)
  - Quality Score Integration: Identity layer for base quality incorporation
- **Input:**
  - Tokenized DNA sequences (vocabulary size: 12)
  - Base quality scores
- **Output:** Per-base classification (artifact vs. non-artifact)

## Uses

### Direct Use

DeepChopper is designed for:
- Detecting chimeric artifacts in Nanopore direct RNA sequencing data
- Identifying adapter sequences within base-called reads
- Preprocessing RNA-seq data before downstream transcriptomics analysis
- Improving accuracy of transcript annotation and gene fusion detection

### Downstream Use

The cleaned data can be used for:
- Transcript isoform analysis
- Gene expression quantification
- Novel transcript discovery
- Gene fusion detection
- Alternative splicing analysis

### Out-of-Scope Use

This model is NOT designed for:
- DNA sequencing data (it's specifically trained on RNA sequences)
- PacBio or Illumina sequencing platforms
- Genome assembly or variant calling

## Training Details

### Training Data

The model was trained on Nanopore direct RNA sequencing data with manually curated annotations of chimeric artifacts and adapter sequences.

### Training Procedure

- **Optimizer:** Adam (lr=0.0002, weight_decay=0)
- **Learning Rate Scheduler:** ReduceLROnPlateau (mode=min, factor=0.1, patience=10)
- **Loss Function:** Continuous Interval Loss (CrossEntropyLoss with no penalty)
- **Framework:** PyTorch Lightning

### Training Hyperparameters

- Learning Rate: 0.0002
- Batch Size: Configured per experiment
- Weight Decay: 0
- Backbone: Fine-tuned (not frozen)

## Evaluation

### Testing Data & Metrics

"""

    if f1_score is not None:
        model_card += f"- **F1 Score:** {f1_score:.4f}\n"

    model_card += """
The model is evaluated on held-out test sets using:
- F1 Score (primary metric)
- Precision
- Recall

### Results

DeepChopper significantly improves downstream analysis quality by accurately removing chimeric artifacts that would otherwise confound transcriptome analyses.

## How to Use

### Installation

```bash
pip install deepchopper
```

### Python API

```python
import deepchopper

# Load the pretrained model
model = deepchopper.DeepChopper.from_pretrained("MODEL_NAME")

# The model is ready for inference
# Use with deepchopper's predict pipeline
```

### Command Line Interface

```bash
# Step 1: Encode your FASTQ data
deepchopper encode input.fq

# Step 2: Predict chimeric artifacts
deepchopper predict input.parquet --output predictions

# Step 3: Remove artifacts and generate clean FASTQ
deepchopper chop predictions input.fq
```

For GPU acceleration:
```bash
deepchopper predict input.parquet --output predictions --gpus 1
```

### Web Interface

Try DeepChopper online without installation:
- [Hugging Face Space](https://huggingface.co/spaces/yangliz5/deepchopper)
- Or run locally: `deepchopper web`

## Limitations

- **Platform-specific:** Optimized for Nanopore direct RNA sequencing
- **Read length:** Best performance on reads up to 32k bases (model context window)
- **Species:** Trained primarily on human RNA sequences
- **Computational requirements:** GPU recommended for large datasets

## Citation

If you use DeepChopper in your research, please cite:

```bibtex
@article{Li2024.10.23.619929,
    author = {Li, Yangyang and Wang, Ting-You and Guo, Qingxiang and Ren, Yanan and Lu, Xiaotong and Cao, Qi and Yang, Rendong},
    title = {A Genomic Language Model for Chimera Artifact Detection in Nanopore Direct RNA Sequencing},
    year = {2024},
    doi = {10.1101/2024.10.23.619929},
    journal = {bioRxiv}
}
```

## Contact & Support

- **Issues:** [GitHub Issues](https://github.com/ylab-hi/DeepChopper/issues)
- **Documentation:** [Full Tutorial](https://github.com/ylab-hi/DeepChopper/blob/main/documentation/tutorial.md)
- **Repository:** [GitHub](https://github.com/ylab-hi/DeepChopper)

## Model Card Authors

YLab Team

## Model Card Contact

For questions about this model, please open an issue on the [GitHub repository](https://github.com/ylab-hi/DeepChopper/issues).
"""

    return model_card.replace("MODEL_NAME", model_name)


@app.command()
def main(
    ckpt_path: str = typer.Argument(..., help="Path to the checkpoint file (.ckpt)"),
    model_name: str = typer.Option(
        "yangliz5/deepchopper",
        "--model-name", "-m",
        help="Model name/identifier on Hugging Face Hub (format: username/model-name)"
    ),
    f1_score: Optional[float] = typer.Option(
        None,
        "--f1-score", "-f",
        help="F1 score to include in model card (auto-detected from filename if not provided)"
    ),
    description: Optional[str] = typer.Option(
        None,
        "--description", "-d",
        help="Custom model description for the model card"
    ),
    commit_message: str = typer.Option(
        "Upload DeepChopper model",
        "--commit-message", "-c",
        help="Commit message for the Hugging Face Hub"
    ),
    private: bool = typer.Option(
        False,
        "--private", "-p",
        help="Make the model repository private"
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token", "-t",
        help="Hugging Face API token (uses HF_TOKEN environment variable if not provided)"
    ),
) -> None:
    """
    Push a DeepChopper checkpoint to Hugging Face Hub with comprehensive metadata.

    This script:
    - Validates the checkpoint file
    - Loads the model from the checkpoint
    - Generates a comprehensive model card
    - Pushes the model to Hugging Face Hub with proper tags and metadata

    Example usage:

        python modle2hub.py epoch_012_f1_0.9947.ckpt --model-name username/deepchopper-v1

        python modle2hub.py model.ckpt -m username/my-model -f 0.9947 --private
    """
    # Validate checkpoint path
    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        console.print(f"[bold red]Error:[/bold red] Checkpoint file not found: {ckpt_path}")
        raise typer.Exit(code=1)

    if not ckpt_file.suffix == ".ckpt":
        console.print(f"[bold yellow]Warning:[/bold yellow] File doesn't have .ckpt extension: {ckpt_path}")

    # Display configuration
    console.print(Panel.fit(
        f"[bold cyan]DeepChopper Model Upload Configuration[/bold cyan]\n\n"
        f"[bold]Checkpoint:[/bold] {ckpt_path}\n"
        f"[bold]Model Name:[/bold] {model_name}\n"
        f"[bold]F1 Score:[/bold] {f1_score if f1_score else 'Auto-detect from filename'}\n"
        f"[bold]Private:[/bold] {private}\n"
        f"[bold]Commit Message:[/bold] {commit_message}",
        title="Upload Configuration",
        border_style="cyan"
    ))

    # Generate model card
    console.print("\n[bold cyan]Generating model card...[/bold cyan]")
    model_card_content = create_model_card(model_name, ckpt_path, f1_score, description)

    # Show model card preview
    if typer.confirm("\nWould you like to preview the model card?", default=False):
        console.print(Panel(model_card_content, title="Model Card Preview", border_style="green"))

    # Confirm upload
    if not typer.confirm("\nProceed with upload to Hugging Face Hub?", default=True):
        console.print("[yellow]Upload cancelled.[/yellow]")
        raise typer.Exit(code=0)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Load model
            task = progress.add_task("Loading model from checkpoint...", total=None)
            model = deepchopper.DeepChopper.from_checkpoint(ckpt_path)  # type: ignore[attr-defined]
            progress.update(task, completed=True)

            # Prepare token
            hf_token = token if token else os.environ.get("HF_TOKEN")

            # Push model to hub
            task = progress.add_task("Pushing model to Hugging Face Hub...", total=None)

            # Prepare push_to_hub kwargs
            push_kwargs = {
                "repo_id": model_name,
                "commit_message": commit_message,
                "private": private,
            }

            if hf_token:
                push_kwargs["token"] = hf_token

            model.push_to_hub(**push_kwargs)
            progress.update(task, completed=True)

            # Upload model card
            task = progress.add_task("Uploading model card (README_HG.md)...", total=None)
            api = HfApi(token=hf_token)

            # Create temporary model card file
            model_card_path = Path("README_HG.md")
            with open(model_card_path, "w", encoding="utf-8") as f:
                f.write(model_card_content)

            # Upload the README.md file
            api.upload_file(
                path_or_fileobj=str(model_card_path),
                path_in_repo="README_HG.md",
                repo_id=model_name,
                commit_message=f"{commit_message} - Add model card",
                repo_type="model",
            )
            progress.update(task, completed=True)

            # Clean up temporary model card
            if model_card_path.exists():
                model_card_path.unlink()

        # Success message
        console.print(Panel.fit(
            f"[bold green]✓ Success![/bold green]\n\n"
            f"Model successfully pushed to: [bold cyan]https://huggingface.co/{model_name}[/bold cyan]\n\n"
            f"To use this model:\n"
            f"[dim]import deepchopper\n"
            f"model = deepchopper.DeepChopper.from_pretrained('{model_name}')[/dim]",
            title="Upload Complete",
            border_style="green"
        ))

    except Exception as e:
        console.print(f"\n[bold red]Error during upload:[/bold red] {str(e)}")
        console.print("\n[yellow]Troubleshooting tips:[/yellow]")
        console.print("1. Make sure you're logged in: [dim]huggingface-cli login[/dim]")
        console.print("2. Check your internet connection")
        console.print("3. Verify the model name format: [dim]username/model-name[/dim]")
        console.print("4. Ensure you have write permissions for this repository")

        # Clean up temporary files
        readme_path = Path("README_HG.md")
        if readme_path.exists():
            readme_path.unlink()

        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
