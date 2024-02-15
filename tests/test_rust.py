
from deepchopper import encode_fqs


def test_encode_fqs():
    data  = "tests/data/test.fq.gz"
    k  = 3
    bases = "ACGTN"
    qual_offset = 33

    inputs, target, kmer2idx = encode_fqs(data, k, bases, qual_offset)

    print(inputs)
    print(target)
    print(kmer2idx)
