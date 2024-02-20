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
