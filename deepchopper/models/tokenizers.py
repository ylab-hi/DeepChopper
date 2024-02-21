from tokenizers import NormalizedString, PreTokenizedString

import deepchopper


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


class KmerPreTokenizer:
    def __init__(self, kmer_size: int, *, overlap: bool):
        self.kmer_size = kmer_size
        self.overlap = overlap

    def kmer_split(self, i: int, normalized_string: NormalizedString) -> list[NormalizedString]:
        return [
            normalized_string[start:end]
            for (_token, (start, end)) in deepchopper.seq_to_kmers_and_offset(
                sequence, self.kmer_size, self.overlap
            )
        ]

    def pre_tokenize(self, pretok: PreTokenizedString):
        # Let's call split on the PreTokenizedString to split using `self.jieba_split`
        pretok.split(self.kmer_split)


class KmerDecoder:
    def decode(self, tokens: List[str]) -> str:
        return "".join(tokens)
