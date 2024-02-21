import deepchopper


class KmerPreTokenizer:
    def __init__(self, kmer_size: int, *, overlap: bool):
        self.kmer_size = kmer_size
        self.overlap = overlap

    def pre_tokenize_str(self, sequence: str) -> list[tuple[str, tuple[int, int]]]:
        """Pre-tokenize a sequence into overlapping kmers.

        Example:
            sequence = "ATCGG"
            kmer_size = 3
            overlap = False
            pre_tokenize_str(sequence) -> [("ATC", (0, 3)), ("TCG", (1, 4)), ("CGG", (2, 5))]
        """
        return deepchopper.seq_to_kmers_and_offset(sequence, self.kmer_size, self.overlap)


def compute_tokens_to_ids(kmer_size: int) -> tuple[dict[str, int], list[str]]:
    kmers_tokens = deepchopper.generate_kmers(deepchopper.default.BASES, kmer_size)
    standard_tokens = kmers_tokens

    unknown_token = "<UNK>"
    padding_token = "<PAD>"
    eos_token = "<EOS>"
    bos_token = "<BOS>"
    sep_token = "<SEP>"

    specical_tokens = [padding_token, unknown_token, eos_token, bos_token, sep_token]
    all_tokens = standard_tokens + specical_tokens
    tokens_to_ids = {tok: i for i, tok in enumerate(all_tokens)}
    return tokens_to_ids, all_tokens
