use anyhow::Result;
use std::path::Path;

use needletail::{parse_fastx_file, FastxReader, Sequence};

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

        let qual = seqrec.qual().unwrap_or(b"");

        println!(">{}", String::from_utf8_lossy(seqrec.id()));
        println!("seq len: {}", seqrec.num_bases());
        println!("{}", String::from_utf8_lossy(qual));

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
    fn test_fq() {
        let fastq_path = PathBuf::from("tests/data/simple.fq");
        let _result = read_fx(fastq_path);
    }
}
