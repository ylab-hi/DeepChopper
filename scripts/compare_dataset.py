from datasets import load_dataset
from datasets import Dataset as HuggingFaceDataset
from deepchopper.data.only_fq import parse_fastq_file

def main():
    parquet_dataset = load_dataset("parquet", data_files={"predict": "./tmp/raw_no_trim.parquet"})["predict"]
    iter_dataset = {i["id"]: i for i in HuggingFaceDataset.from_generator(parse_fastq_file, gen_kwargs={"file_path": "./tmp/raw_no_trim.fastq", "has_targets": False}).with_format("torch")}


    for key, value in parquet_dataset.items():
        print(f"comparing {key}")

        if key not in iter_dataset:
            print(f"{key} not in iter_dataset")
            break
        if iter_dataset[key]["seq"] != value["seq"]:
            print(f"{key} seq not match")
            break

        if iter_dataset[key]["qual"] != value["qual"]:
            print(f"{key} qual not match")
            break



if __name__ == "__main__":
    main()
