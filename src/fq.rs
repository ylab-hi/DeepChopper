use anyhow::{anyhow, Result};
use derive_builder::Builder;
use std::path::Path;

use bio::utils::Interval; // 0-based, half-open interval [1, 10)
use needletail::{parse_fastx_file, parser::SequenceRecord, FastxReader, Sequence};

// @462:528|738735b7-2105-460e-9e56-da980ef816c2+4f605fb4-4107-4827-9aed-9448d02834a8
// CGTTGGTGGTGTTCAGTTGTGGCGGTTGCTGGTCAGTAACAGCCAAGATGCTGCGGAATCTGCTGGCTTACCGTCAGATTGGGCAGAGGACGATAAGCACTGCTTCCCGCAGGCATTTTAAAAATAAAGTTCCGGAGAAGCAAAACTGTTCCAGGAGGATGATGAAATTCCACTGTATCTAAAAGGGTAGGGTAGCTGATGCCCTCCTGTATAGAGCCACCATGATCTTACAGTTGGTGGAACAGCATATGCCATATATGAGCTGGCTGTGGCTTCATTTCCCAAGAAGCAGGAGTGACTTTCAGCTTTATCTCCAGCAATTGCTTGGTCAGTTTTTCATTCAGCTCTCTATGGACCAGTAATCTGATAAATAACCGAGCTCTTCTTTGGGGATCAATATTTATTGATTGTAGTAACTGCCACCAATAAAGCAGTCTTTACCATGAAAAAAAAAAAAAAAAATCCCCCTACCCCTCTCTCCCAACTTATCCATACACAACCTGCCCCTCCAACCTCTTTCTAAACCCTTGGCGCCTCGGAGGCGTTCAGCTGCTTCAAGATGAAGCTGAACATCTTCCTTCCCAGCCACTGGCTGCCAGAAACTCATTGAAGTGGACGATGAACGCAAACTTCTGCACTTTCTATGAGAAGCGTATGGCCACAGAAGTTGCTGCTGACGCTTTGGGTGAAGAATGGAAGGGTTATGTGGTCCGAATCAGTGGTGGGAACGACAAACAAGGTTTCCCCATGAAGCAGGGTGTCTTGACCCATGGCCGTGTCCGCCTGCTACTGAGTAAGGGGCATTCCTGTTACAGACCAAGGAGAACTGGAGAAAGAAAGAGAAAATCAGTTCGTGGTTGCATTGTGGATGCAATCTGAGCGTTCTCAACTTGGTTATTGTAAAAAGGAGAGAAGGATATTCCTGGACTGACTGATACTACAGTGCCTCGCCGCCTGGGCCCCAAAAGAGCTAGCAGAATCCGCAAACTTTTCAATCTCTCTAAAAAGAAGATGATGTCCGCCAGTATCGTTGTAAGAAAGCCCTAAAATAAAGAAGGTAAGAAACCTAGGACCAAAGCACCCAAGATTCAGCGTCTGTTACTCCACGTGTCCTGCAGCACAAACGGCGGCGTATTGCTCTGAAGAAGCAGCGTACCAAGAAAAATAAAAGAAGAGGCTGCAGAATATGCTAAACTTTTGGCCTAGAGAATGAAGGAGGCTAAGGAGAAGCGCCAGGAACAAATTGCGAAGAGACGCAGACTTTCCTCTCTGCGGGACTCTACTTCTAAGTCTGAATCCAGTCAGAAATAAGATTTTTTGAGTAACAAATAATAAGATCGGGACTCTGA
// +
// 3A88;IMSJ872377DIJRSRRQRSSRSSSSSSSSGECCKLDIDDGLQRSRRRROSSPRRNOOSSEBCDEJSQKHHJSSSSSSSMMMPSSSSSRJ97677;<SSSSSRRSSSSSSSSSSSSJJKSSSSSSHFFBFGLBCBC<OPLMOP?KSIII6435@ESSSSKSSSSPSSSSD?22275@DB;(((478GIIJKMSIFEFKFA2-)&&''=ALPQQSSRSS,,;>SSSSSSSSSSSSOKGKOSSSSSSSQFLHGISSSSIGGHSSSSFFB.AGA0<AKLM9SSPLLMKMKLJJ..-02::<=0,+-)&&-:?BEFHNLIBA>>E89SSSSSASSQPSOPLMHG7788SSSCB==BCKLMPPPQQKIINRSSSSSSSSSSSSSSSSSPSRPOPPGGCH,,CEH1*%289<DACCRGGHISSSSSSSSQRQRSSSSSMQSRRRFFPPPPPPPPPPPPPPPPP--'.,,/42$))(('')'0314319HFF2104/)*+&#""33%%%(%%$##"""""""&..05%$((*(*36FSSSSSSS8794555HJI0///?SSSSSSSSSSSSSFD110AHKHKKJMJNOPS@;@@@HMQSSLMSOFC>546:<JNSIIIIKJJKSSSSSROO+(((--,1BLJKKSSSSSSSSSSSSSPKJLPSSSSSSSSSMFGPS22116559IIIISQQSSSSSSSSSSSSSQPMB651.13SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSCC>.1ABR+(96;>SSRSQPPPSSSSSSRQMQNQRSSSSSSSSSSMSSSSSMKICDSRQPIF>>>?CL0GHLNMC=<;DBCCBBBCDABBSSSSSRRSSQRMNPPROHGBCFBBBAGGGL@OEC>53038=JSSSROJGKIIHGDDFLOOOSSIBCENLNENFFSSSSNMLF@JSSSPPRQR;:989FLPDBA00//7>PSSSSSSMIHHIRSSJGGIKKPPRRSED;:;?JO=::EKIGBD'&),QQSSSN>S/0KJJIMNORLH@6679FCF5556OOORNNLRQSSPSIII>4HF6A?=AHHSSPOKSLSN-,-?EFMSRSQ;;;DIIGEEIJOSLFE@?*)+-<CD?CFDA999AK;HIFGMQSSSSSSOKCBDSSSOLJ43115AJJGKKLOOMMNLOQSSSSI1,1CCOSS11:IIMSSSSSSSSSSSSOQRRSNJHSSSSSL/../<DEKMLLGPNOA?>>MOSSLKSSSBBCJRRRQSSSSS>76654;<BSONKIKK.--+0,,51+)+450045-,.5OSSHED777SSEEEJSSSSKKIGOOSSSSIIHJSSSSSSLIJOQSSJMSS;EFAA5**)+-2556BBJOM

#[derive(Debug, Builder, Default)]
pub struct FqEncoderOption {
    #[builder(default = "3")]
    kmer_size: usize,
}

pub struct FqEncoder {
    option: FqEncoderOption,
}

impl FqEncoder {
    pub fn new(option: FqEncoderOption) -> Self {
        Self { option }
    }

    fn parse_target_from_id(src: &[u8]) -> Result<Vec<Interval<usize>>> {
        // check empty input
        if src.is_empty() {
            return Ok(Vec::new());
        }

        // @462:528,100:120
        let mut targets = Vec::new();

        // remove the leading '@' character
        let src = &src[1..];

        for target in src.split(|&c| c == b',') {
            let mut parts = target.split(|&c| c == b':');
            let start: usize = lexical::parse(parts.next().ok_or(anyhow!("parse number error"))?)?;
            let end: usize = lexical::parse(parts.next().ok_or(anyhow!("parse number error"))?)?;
            targets.push(Interval::new(start..end)?);
        }
        Ok(targets)
    }

    fn encode_qual(&self, qual: &[u8], socre_offset: Option<u8>) -> Vec<u8> {
        let offset = socre_offset.unwrap_or(33);

        // input is quality of fastq
        // 1. convert the quality to a score
        // 2. return the score
        qual.iter()
            .map(|&q| {
                // Convert ASCII to Phred score for Phred+33 encoding
                q - offset
            })
            .collect()
    }

    fn encoder_seq(&self, seq: &[u8]) -> Vec<Vec<u8>> {
        // 1. normalize the sequence
        // 2. make a reverse complemented copy of the sequence first for
        // `canonical_kmers` to draw the complemented sequences from.
        // 3. return the canonical kmers
        unimplemented!()
    }

    pub fn encode_fq(&self, record: &SequenceRecord) {
        // 1.encode the sequence
        // 2. encode the quality
    }
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
            println!("{}", String::from_utf8_lossy(kmer));
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

    #[test]
    fn test_parse_target_from_id() {
        // Test case 1: Valid input
        let src = b"@462:528,100:120";
        let expected = vec![
            Interval::new(462..528).unwrap(),
            Interval::new(100..120).unwrap(),
        ];

        assert_eq!(parse_target_from_id(src).unwrap(), expected);

        // Test case 2: Empty input
        let src = b"";
        let expected: Vec<Interval<usize>> = Vec::new();
        assert_eq!(parse_target_from_id(src).unwrap(), expected);

        // Test case 3: Invalid input (missing colon)
        let src = b"@462528,100:120";
        assert!(parse_target_from_id(src).is_err());

        // Test case 4: Invalid input (invalid number)
        let src = b"@462:528,100:abc";
        assert!(parse_target_from_id(src).is_err());
    }
}
