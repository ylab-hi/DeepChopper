# Tutorial: Using DeepChopper for Nanopore Direct-RNA Sequencing Data Analysis

This tutorial will guide you through the process of using DeepChopper to identify and remove chimeric artificial reads in Nanopore direct-RNA sequencing data. We'll cover each step from data acquisition to the final chopping of chimeric reads.

## Table of Contents

- [Tutorial: Using DeepChopper for Nanopore Direct-RNA Sequencing Data Analysis](#tutorial-using-deepchopper-for-nanopore-direct-rna-sequencing-data-analysis)
	- [Table of Contents](#table-of-contents)
	- [1. Get the Data](#1-get-the-data)
	- [2. Basecall Using Dorado](#2-basecall-using-dorado)
	- [3. DeepChopper Encode](#3-deepchopper-encode)
	- [4. DeepChopper Predict](#4-deepchopper-predict)
	- [5. DeepChopper Chop](#5-deepchopper-chop)
	- [Conclusion](#conclusion)

## 1. Get the Data

First, you need to obtain your Nanopore direct-RNA sequencing data. This data is typically in the form of FAST5 files.

```bash
# Example: Download sample data (replace with your actual data source)
wget https://example.com/sample_nanopore_data.fast5
```

Ensure you have sufficient storage space, as Nanopore data can be quite large.

## 2. Basecall Using Dorado

Next, we'll use Dorado, Oxford Nanopore's high-performance basecaller, to convert the raw signal data into nucleotide sequences. 
It's important to run Dorado without the trimming option to preserve potential chimeric sequences for DeepChopper to analyze.

```bash
# Install Dorado (if not already installed)
conda install -c bioconda ont-dorado

# Run Dorado without trimming
dorado basecaller \
    --model dna_r10.4.1_e8.2_400bps_sup@v4.2.0 \
    --device cuda:all \
    --not_trim \
    path/to/your/fast5/files/ \
    > output.fastq
```

Replace `path/to/your/fast5/files/` with the directory containing your FAST5 files. 
The output will be a FASTQ file containing the basecalled sequences.

## 3. DeepChopper Encode

Now that we have our basecalled sequences, we'll use DeepChopper to encode the data. 
This step prepares the data for the prediction model.

```bash
# Encode the FASTQ file
deepchopper encode output.fastq -o encoded_data.parquet
```

This command will generate a Parquet file (`encoded_data.parquet`) containing the encoded sequences.

## 4. DeepChopper Predict

With our encoded data, we can now use DeepChopper to predict chimeric reads.

```bash
# Predict chimeric reads
deepchopper predict encoded_data.parquet -o predictions.csv
```

This step will analyze the encoded data and produce a CSV file (`predictions.csv`) containing predictions for each read, indicating whether it's likely to be chimeric or not.

## 5. DeepChopper Chop

Finally, we'll use DeepChopper to chop the identified chimeric reads, removing artificial sequences and preserving the genuine RNA sequences.

```bash
# Chop chimeric reads
deepchopper chop output.fastq predictions.csv -o cleaned_output.fastq
```

This command takes the original FASTQ file (`output.fastq`) and the predictions (`predictions.csv`), and produces a new FASTQ file (`cleaned_output.fastq`) with the chimeric reads chopped.

## Conclusion

You've now successfully processed your Nanopore direct-RNA sequencing data using DeepChopper! The `cleaned_output.fastq` file contains your sequencing data with chimeric artificial reads identified and removed, providing you with higher quality data for your downstream analyses.

Remember to adjust file paths and names according to your specific setup and data.
For more advanced usage and options, refer to the DeepChopper documentation or use the `--help` flag with each command.

Happy sequencing!