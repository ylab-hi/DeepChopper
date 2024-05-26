"""Command genes count from isoquant.

@author: YangyangLi
@contact: yangyang.li@northwestern.edu
@time: 2024-05-23
"""

import sys
import logging
from rich.logging import RichHandler

from pathlib import Path


FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

logger = logging.getLogger("__name__")


def load_genes_count(file: Path):
    genes_count = {}
    file = Path(file)
    file_handle = file.open()
    next(file_handle)

    for line in file_handle:
        line_content = line.strip().split("\t")
        gene = line_content[0]
        count = float(line_content[1])
        if count > 0:
            genes_count[gene] = count

    return genes_count


def main(genecount1: Path, genecount2: Path):
    genes_count1 = load_genes_count(genecount1)
    genes_count2 = load_genes_count(genecount2)

    genes1 = set(genes_count1.keys())
    genes2 = set(genes_count2.keys())

    genes1_only = genes1 - genes2
    genes2_only = genes2 - genes1

    logger.info(f"Genes only in {genecount1}: {len(genes1_only)}")
    logger.info(f"Genes only in {genecount2}: {len(genes2_only)}")

    with Path("genes1_only.txt").open("w") as f:
        for gene in genes1_only:
            f.write(f"{gene}\t{genes_count1[gene]}\n")

    with Path("genes2_only.txt").open("w") as f:
        for gene in genes2_only:
            f.write(f"{gene}\t{genes_count2[gene]}\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.info("Usage: python compare_genes.py <gene_count1> <gene_count2>")
        sys.exit(1)

    genecount1 = Path(sys.argv[1])
    genecount2 = Path(sys.argv[2])

    logger.info(f"Comparing {genecount1} and {genecount2}")
    main(genecount1, genecount2)
