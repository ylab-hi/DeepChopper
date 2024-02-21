from deepchopper.models import KmerPreTokenizer

def test_pre_tokenize_str_no_overlap():
    tokenizer = KmerPreTokenizer(3, overlap=False)
    sequence = "ATCGGCC"
    expected_output = [("ATC", (0, 3)), ("GGC", (3, 6)) ]
    res= tokenizer.pre_tokenize_str(sequence)
    assert res == expected_output



def test_pre_tokenize_str_with_overlap():
    tokenizer = KmerPreTokenizer(3, overlap=True)
    sequence = "ATCGG"
    expected_output = [('ATC', (0, 3)), ('TCG', (1, 4)), ('CGG', (2, 5))]
    res = tokenizer.pre_tokenize_str(sequence)
    assert res == expected_output
