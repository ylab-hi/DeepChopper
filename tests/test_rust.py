from deepchopper import encode_fq_path_to_parquet,encode_fq_path_to_tensor
from pathlib import Path



def test_encode_fqs_to_parquet(tmp_path):
    data  = Path("tests/data/twenty_five_records.fq")
    k  = 3
    bases = "ACGTN"
    qual_offset = 33
    result_path = tmp_path / "result.parquet"

    encode_fq_path_to_parquet(data, k, bases, qual_offset, False, result_path=result_path)
    import pyarrow.parquet as pq

    df = pq.read_table(result_path)
    df_pd = df.to_pandas()
    assert df_pd.shape == (25, 4)



def test_encode_fqs_to_tensor():
    data  = Path("tests/data/one_record.fq")
    k  = 3
    bases = "ACGTN"
    qual_offset = 33

    inputs, target, qual, kmer2idx = encode_fq_path_to_tensor(data, k, bases, qual_offset, True)

    print(f"DBG-YYL[3]: test_rust.py:10: shape: {inputs.shape} inputs={inputs}")
    print(f"DBG-YYL[3]: test_rust.py:11: shape: {target.shape} target={target}")
    print(f"DBG-YYL[3]: test_rust.py:12: shape: {qual.shape} qual={qual}")
    print(f"DBG-YYL[3]: test_rust.py:13: kmer2idx={kmer2idx}")
