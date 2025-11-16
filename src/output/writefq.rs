use std::fs::File;
use std::io::{BufReader, Read};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::{io, thread};

use anyhow::Result;
use noodles::fastq::{self as fastq, record::Definition};

use flate2::read::GzDecoder;
use noodles::bgzf;
use noodles::fastq::record::Record as FastqRecord;
use rayon::prelude::*;

use crate::fq_encode::RecordData;

pub fn write_fq(records: &[RecordData], file_path: Option<PathBuf>) -> Result<()> {
    let sink: Box<dyn io::Write> = if let Some(file) = file_path {
        Box::new(File::create(file)?)
    } else {
        Box::new(io::stdout().lock())
    };
    let mut writer = fastq::io::Writer::new(sink);

    for record in records {
        let qual_str = record.qual.to_string();

        let record = fastq::Record::new(
            Definition::new(record.id.to_vec(), ""),
            record.seq.to_vec(),
            qual_str,
        );
        writer.write_record(&record)?;
    }

    Ok(())
}

/// Represents different types of file compression formats
///
/// This enum is used to identify and handle various compression formats commonly used for files.
/// It can be used in Python through the deepbiop.utils module.
///
/// # Variants
///
/// * `Uncompress` - Uncompressed/raw file format
/// * `Gzip` - Standard gzip compression (.gz files)
/// * `Bgzip` - Blocked gzip format, commonly used in bioinformatics
/// * `Zip` - ZIP archive format
/// * `Bzip2` - bzip2 compression format
/// * `Xz` - XZ compression format (LZMA2)
/// * `Zstd` - Zstandard compression format
/// * `Unknown` - Unknown or unrecognized compression format
#[derive(Debug, PartialEq, Clone, Eq, Hash)]
pub enum CompressedType {
    Uncompress,
    Gzip,
    Bgzip,
    Zip,
    Bzip2,
    Xz,
    Zstd,
    Unknown,
}

/// Determines the compression type of a file by examining its header/signature
///
/// This function reads the first few bytes of a file and checks for known magic numbers
/// or file signatures to identify the compression format used.
///
/// # Arguments
///
/// * `file_path` - Path to the file to check, can be any type that converts to a Path
///
/// # Returns
///
/// * `Result<CompressedType>` - The detected compression type wrapped in a Result
///
/// # Errors
///
/// Returns an error if:
/// * The file cannot be opened
/// * There are issues reading the file header
pub fn check_compressed_type<P: AsRef<Path>>(file_path: P) -> Result<CompressedType> {
    let mut file = File::open(file_path)?;
    let mut buffer = [0u8; 18]; // Large enough for BGZF detection

    // Read the first few bytes
    let bytes_read = file.read(&mut buffer)?;
    if bytes_read < 2 {
        return Ok(CompressedType::Uncompress);
    }

    // Check magic numbers/file signatures
    match &buffer[..] {
        // Check for BGZF first (starts with gzip magic number + specific extra fields)
        [0x1f, 0x8b, 0x08, 0x04, ..] if bytes_read >= 18 => {
            // Check for BGZF extra field
            let xlen = u16::from_le_bytes([buffer[10], buffer[11]]) as usize;
            if xlen >= 6 && buffer[12] == 0x42  // B
                && buffer[13] == 0x43  // C
                && buffer[14] == 0x02  // Length of subfield (2)
                && buffer[15] == 0x00
            // Length of subfield (2)
            {
                Ok(CompressedType::Bgzip)
            } else {
                Ok(CompressedType::Gzip)
            }
        }

        // Regular Gzip: starts with 0x1F 0x8B
        [0x1f, 0x8b, ..] => Ok(CompressedType::Gzip),

        // Zip: starts with "PK\x03\x04" or "PK\x05\x06" (empty archive) or "PK\x07\x08" (spanned archive)
        [0x50, 0x4b, 0x03, 0x04, ..]
        | [0x50, 0x4b, 0x05, 0x06, ..]
        | [0x50, 0x4b, 0x07, 0x08, ..] => Ok(CompressedType::Zip),

        // Bzip2: starts with "BZh"
        [0x42, 0x5a, 0x68, ..] => Ok(CompressedType::Bzip2),

        // XZ: starts with 0xFD "7zXZ"
        [0xfd, 0x37, 0x7a, 0x58, 0x5a, 0x00, ..] => Ok(CompressedType::Xz),

        // Zstandard: starts with magic number 0xFD2FB528
        [0x28, 0xb5, 0x2f, 0xfd, ..] => Ok(CompressedType::Zstd),

        // If no compression signature is found, assume it's a normal file
        _ => {
            // Additional check for text/binary content could be added here
            Ok(CompressedType::Uncompress)
        }
    }
}

/// Creates a reader for a file that may be compressed
///
/// This function detects the compression type of the file and returns an appropriate reader.
/// Currently supports uncompressed files, gzip, and bgzip formats.
///
/// # Arguments
/// * `file_path` - Path to the file to read, can be compressed or uncompressed
///
/// # Returns
/// * `Ok(Box<dyn io::Read>)` - A boxed reader appropriate for the file's compression
/// * `Err` - If the file cannot be opened or has an unsupported compression type
pub fn create_reader_for_compressed_file<P: AsRef<Path>>(
    file_path: P,
) -> Result<Box<dyn io::Read>> {
    let compressed_type = check_compressed_type(file_path.as_ref())?;
    let file = File::open(file_path)?;

    Ok(match compressed_type {
        CompressedType::Uncompress => Box::new(file),
        CompressedType::Gzip => Box::new(GzDecoder::new(file)),
        CompressedType::Bgzip => Box::new(bgzf::io::Reader::new(file)),
        _ => return Err(anyhow::anyhow!("unsupported compression type")),
    })
}

pub fn read_noodel_records_from_fq_or_zip_fq<P: AsRef<Path>>(
    file_path: P,
) -> Result<Vec<FastqRecord>> {
    let reader = create_reader_for_compressed_file(&file_path)?;
    let mut reader = fastq::io::Reader::new(BufReader::new(reader));
    reader.records().map(|record| Ok(record?)).collect()
}

/// Streaming reader for FASTQ records that owns the underlying reader
///
/// This struct provides an iterator interface for reading FASTQ records
/// without loading all records into memory at once.
pub struct StreamingFastqReader {
    reader: fastq::io::Reader<BufReader<Box<dyn io::Read>>>,
}

impl StreamingFastqReader {
    /// Creates a new streaming reader for a FASTQ file (compressed or uncompressed)
    pub fn new<P: AsRef<Path>>(file_path: P) -> Result<Self> {
        let reader = create_reader_for_compressed_file(&file_path)?;
        let reader = fastq::io::Reader::new(BufReader::new(reader));
        Ok(Self { reader })
    }
}

impl Iterator for StreamingFastqReader {
    type Item = Result<FastqRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        self.reader.records().next().map(|r| r.map_err(Into::into))
    }
}

pub fn read_noodle_records_from_fq<P: AsRef<Path>>(file_path: P) -> Result<Vec<FastqRecord>> {
    let mut reader = File::open(file_path)
        .map(BufReader::new)
        .map(fastq::io::Reader::new)?;
    let records: Result<Vec<FastqRecord>> = reader
        .records()
        .par_bridge()
        .map(|record| {
            let record = record?;
            Ok(record)
        })
        .collect();
    records
}

pub fn write_fq_for_noodle_record<P: AsRef<Path>>(
    records: &[FastqRecord],
    file_path: P,
) -> Result<()> {
    let file = File::create(file_path)?;
    let mut writer = fastq::io::Writer::new(file);
    for record in records {
        writer.write_record(record)?;
    }
    Ok(())
}

pub fn write_zip_fq_parallel(
    records: &[RecordData],
    file_path: PathBuf,
    threads: Option<usize>,
) -> Result<()> {
    let worker_count = NonZeroUsize::new(threads.unwrap_or(1))
        .map(|count| count.min(thread::available_parallelism().unwrap()))
        .unwrap();

    let sink = File::create(file_path)?;
    let encoder = bgzf::io::MultithreadedWriter::with_worker_count(worker_count, sink);

    let mut writer = fastq::io::Writer::new(encoder);

    for record in records {
        let record = fastq::Record::new(
            Definition::new(record.id.to_vec(), ""),
            record.seq.to_vec(),
            record.qual.to_vec(),
        );
        writer.write_record(&record)?;
    }
    Ok(())
}

pub fn write_fq_parallel_for_noodle_record(
    records: &[FastqRecord],
    file_path: PathBuf,
    threads: Option<usize>,
) -> Result<()> {
    let worker_count = NonZeroUsize::new(threads.unwrap_or(2))
        .map(|count| count.min(thread::available_parallelism().unwrap()))
        .unwrap();

    let sink = File::create(file_path)?;
    let encoder = bgzf::io::MultithreadedWriter::with_worker_count(worker_count, sink);

    let mut writer = fastq::io::Writer::new(encoder);

    for record in records {
        writer.write_record(record)?;
    }
    Ok(())
}

pub fn read_noodle_records_from_gzip_fq<P: AsRef<Path>>(file_path: P) -> Result<Vec<FastqRecord>> {
    let mut reader = File::open(file_path)
        .map(GzDecoder::new)
        .map(BufReader::new)
        .map(fastq::io::Reader::new)?;

    let records: Result<Vec<FastqRecord>> = reader
        .records()
        .par_bridge()
        .map(|record| {
            let record = record?;
            Ok(record)
        })
        .collect();
    records
}

pub fn read_noodle_records_from_bzip_fq<P: AsRef<Path>>(file_path: P) -> Result<Vec<FastqRecord>> {
    let decoder = bgzf::io::Reader::new(File::open(file_path)?);
    let mut reader = fastq::io::Reader::new(decoder);

    let records: Result<Vec<FastqRecord>> = reader
        .records()
        .par_bridge()
        .map(|record| {
            let record = record?;
            Ok(record)
        })
        .collect();
    records
}

pub fn convert_multiple_zip_fqs_to_one_zip_fq<P: AsRef<Path>>(
    paths: &[PathBuf],
    result_path: P,
    parallel: bool,
) -> Result<()> {
    let records = if parallel {
        paths
            .par_iter()
            .flat_map(|path| read_noodle_records_from_bzip_fq(path).unwrap())
            .collect::<Vec<FastqRecord>>()
    } else {
        paths
            .iter()
            .flat_map(|path| read_noodle_records_from_bzip_fq(path).unwrap())
            .collect::<Vec<FastqRecord>>()
    };
    write_fq_parallel_for_noodle_record(&records, result_path.as_ref().to_path_buf(), None)?;
    Ok(())
}

pub fn convert_multiple_fqs_to_one_zip_fq<P: AsRef<Path>>(
    paths: &[PathBuf],
    result_path: P,
    parallel: bool,
) -> Result<()> {
    let records = if parallel {
        paths
            .par_iter()
            .flat_map(|path| read_noodle_records_from_fq(path).unwrap())
            .collect::<Vec<FastqRecord>>()
    } else {
        paths
            .iter()
            .flat_map(|path| read_noodle_records_from_fq(path).unwrap())
            .collect::<Vec<FastqRecord>>()
    };
    write_fq_parallel_for_noodle_record(&records, result_path.as_ref().to_path_buf(), None)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::NamedTempFile;

    #[test]
    fn test_streaming_fastq_reader() {
        // Create test FASTQ records
        let test_records = vec![
            RecordData {
                id: b"read1".into(),
                seq: b"ATCGATCG".into(),
                qual: b"IIIIIIII".into(),
            },
            RecordData {
                id: b"read2".into(),
                seq: b"GCTAGCTA".into(),
                qual: b"HHHHHHHH".into(),
            },
            RecordData {
                id: b"read3".into(),
                seq: b"AAAATTTT".into(),
                qual: b"JJJJJJJJ".into(),
            },
        ];

        // Write test data to a temporary gzip file
        let file = NamedTempFile::new().unwrap();
        let file_path = file.path().to_path_buf();
        write_zip_fq_parallel(&test_records, file_path.clone(), Some(2)).unwrap();

        // Test streaming reader
        let mut streaming_reader = StreamingFastqReader::new(&file_path).unwrap();

        let mut count = 0;
        while let Some(result) = streaming_reader.next() {
            let record = result.unwrap();
            assert!(count < test_records.len());

            let id = record.definition().name();
            let seq = record.sequence();
            let qual = record.quality_scores();

            assert_eq!(id, test_records[count].id.as_slice());
            assert_eq!(seq, test_records[count].seq.as_slice());
            assert_eq!(qual, test_records[count].qual.as_slice());

            count += 1;
        }

        assert_eq!(count, test_records.len());
    }

    #[test]
    fn test_write_fq_with_file_path() {
        let records = vec![
            RecordData {
                id: b"1".into(),
                seq: b"ATCG".into(),
                qual: b"HHHH".into(),
            },
            RecordData {
                id: b"2".into(),
                seq: b"GCTA".into(),
                qual: b"MMMM".into(),
            },
        ];
        let file = NamedTempFile::new().unwrap();
        let file_path = Some(file.path().to_path_buf());

        write_fq(&records, file_path).unwrap();

        let contents = std::fs::read_to_string(file.path()).unwrap();
        assert_eq!(contents, "@1\nATCG\n+\nHHHH\n@2\nGCTA\n+\nMMMM\n");
    }

    #[test]
    fn test_write_fq_parallel() {
        // Create some test data
        let records = vec![
            RecordData {
                id: b"record1".into(),
                seq: b"ATCG".into(),
                qual: b"IIII".into(),
            },
            RecordData {
                id: b"record2".into(),
                seq: b"GCTA".into(),
                qual: b"EEEE".into(),
            },
        ];

        // Create a temporary file to write the records to
        let file = NamedTempFile::new().unwrap();
        let file_path = file.path().to_path_buf();

        // Call the function being tested
        write_zip_fq_parallel(&records, file_path, None).unwrap();

        let decoder = bgzf::io::Reader::new(file.reopen().unwrap());
        let mut reader = fastq::io::Reader::new(decoder);

        let actual_result: Vec<RecordData> = reader
            .records()
            .par_bridge()
            .map(|record| {
                let record = record.unwrap();
                let id = record.definition().name();
                let seq = record.sequence();
                let qual = record.quality_scores();
                RecordData {
                    id: id.into(),
                    seq: seq.into(),
                    qual: qual.into(),
                }
            })
            .collect();

        actual_result.iter().zip(records.iter()).for_each(|(a, b)| {
            assert_eq!(a.id, b.id);
            assert_eq!(a.seq, b.seq);
            assert_eq!(a.qual, b.qual);
        });
    }
}
