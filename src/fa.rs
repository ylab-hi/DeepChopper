use anyhow::Result;
use std::path::Path;

use noodles_fasta as fasta;

use needletail::{parse_fastx_file, FastxReader, Sequence};

pub fn read_fa<P: AsRef<Path>>(fasta_path: P) -> Result<Vec<fasta::record::Record>> {
    let mut reader = fasta::reader::Builder.build_from_path(fasta_path.as_ref())?;
    let results: Vec<_> = reader.records().map(|record| record.unwrap()).collect();
    Ok(results)
}

pub fn read_fx<P: AsRef<Path>>(path: P) -> Result<()> {
    let mut n_bases = 0;
    let mut reader = parse_fastx_file(path.as_ref()).expect("valid path/file");

    while let Some(record) = reader.next() {
        let seqrec = record.expect("invalid record");
        // keep track of the total number of bases
        n_bases += seqrec.num_bases();
        // normalize to make sure all the bases are consistently capitalized and
        // that we remove the newlines since this is FASTA
        let norm_seq = seqrec.normalize(false);
        // we make a reverse complemented copy of the sequence first for
        // `canonical_kmers` to draw the complemented sequences from.
        let rc = norm_seq.reverse_complement();

        println!(">{}", String::from_utf8_lossy(seqrec.id()));
        // now we keep track of the number of AAAAs (or TTTTs via
        // canonicalization) in the file; note we also get the position (i.0;
        // in the event there were `N`-containing kmers that were skipped)
        // and whether the sequence was complemented (i.2) in addition to
        // the canonical kmer (i.1)
        for (_, kmer, _) in norm_seq.canonical_kmers(4, &rc) {
            println!("{}", String::from_utf8_lossy(&kmer));
        }
    }

    println!("Total bases: {}", n_bases);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_read_fx() {
        let fasta_path = PathBuf::from("tests/data/test.fa");
        let result = read_fx(fasta_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_read_fa() {
        let fasta_path = PathBuf::from("tests/data/test.fa");

        // Call the read_fa function
        let result = read_fa(fasta_path);

        // Assert that the result is Ok
        assert!(result.is_ok());

        // Assert that the returned vector has the correct length
        let records = result.unwrap();
        assert_eq!(records.len(), 2);

        // Assert that the first record has the correct name and sequence
        let first_record = &records[0];
        assert_eq!(first_record.name(), b"case1");
        assert_eq!(
            String::from_utf8_lossy(first_record.sequence().as_ref()),
            "AATTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"
        );

        // Assert that the second record has the correct name and sequence
        let second_record = &records[1];
        assert_eq!(second_record.name(), b"case2");
        assert_eq!(
            String::from_utf8_lossy(second_record.sequence().as_ref()),
            "CCTTTAAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGTTTT"
        );
    }
}
