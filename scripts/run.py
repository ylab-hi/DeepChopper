import sys
from pathlib import Path
from deepchopper import encode_fq_path

import numpy as np
import logging

def main(file: Path):

    logging.basicConfig(level=logging.DEBUG)
    data  = file
    k  = 3
    bases = "ACGTN"
    qual_offset = 33
    inputs, target, qual, kmer2idx = encode_fq_path(data, k, bases, qual_offset, True)

    print(f"inputs shape: {inputs.shape}")
    print(f"target shape: {target.shape}")
    print(f"qual shape: {qual.shape}")
    print(f"kmer2idx={kmer2idx}")

    print("save to npz")

    np.savez("inputs.npz", inputs)
    np.savez("target.npz", target)
    np.savez("qual.npz", qual)

    with open("kmer2idx.txt", "w") as f:
        for kmer, idx in kmer2idx.items():
            f.write(f"{kmer}\t{idx}\n")


if __name__ == "__main__":
    file = Path(sys.argv[1])
    main(file)
