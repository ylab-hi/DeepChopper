from deepchopper import encode_fq_path
from pathlib import Path


def test_encode_fqs():
    data  = Path("tests/data/one_record.fq")
    k  = 3
    bases = "ACGTN"
    qual_offset = 33

    inputs, target, qual, kmer2idx = encode_fq_path(data, k, bases, qual_offset, True)

    print(f"DBG-YYL[3]: test_rust.py:10: shape: {inputs.shape} inputs={inputs}")
    print(f"DBG-YYL[3]: test_rust.py:11: shape: {target.shape} target={target}")
    print(f"DBG-YYL[3]: test_rust.py:12: shape: {qual.shape} qual={qual}")
    print(f"DBG-YYL[3]: test_rust.py:13: kmer2idx={kmer2idx}")
