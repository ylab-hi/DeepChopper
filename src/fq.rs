use anyhow::{anyhow, Result};
use derive_builder::Builder;
use std::path::Path;

use bio::utils::Interval; // 0-based, half-open interval [1, 10)
use needletail::{parse_fastx_file, Sequence};
use numpy::ndarray::prelude::*;
use rayon::prelude::*;

use crate::kmer::{generate_kmers_table, KmerTable};

// @462:528|738735b7-2105-460e-9e56-da980ef816c2+4f605fb4-4107-4827-9aed-9448d02834a8
// CGTTGGTGGTGTTCAGTTGTGGCGGTTGCTGGTCAGTAACAGCCAAGATGCTGCGGAATCTGCTGGCTTACCGTCAGATTGGGCAGAGGACGATAAGCACTGCTTCCCGCAGGCATTTTAAAAATAAAGTTCCGGAGAAGCAAAACTGTTCCAGGAGGATGATGAAATTCCACTGTATCTAAAAGGGTAGGGTAGCTGATGCCCTCCTGTATAGAGCCACCATGATCTTACAGTTGGTGGAACAGCATATGCCATATATGAGCTGGCTGTGGCTTCATTTCCCAAGAAGCAGGAGTGACTTTCAGCTTTATCTCCAGCAATTGCTTGGTCAGTTTTTCATTCAGCTCTCTATGGACCAGTAATCTGATAAATAACCGAGCTCTTCTTTGGGGATCAATATTTATTGATTGTAGTAACTGCCACCAATAAAGCAGTCTTTACCATGAAAAAAAAAAAAAAAAATCCCCCTACCCCTCTCTCCCAACTTATCCATACACAACCTGCCCCTCCAACCTCTTTCTAAACCCTTGGCGCCTCGGAGGCGTTCAGCTGCTTCAAGATGAAGCTGAACATCTTCCTTCCCAGCCACTGGCTGCCAGAAACTCATTGAAGTGGACGATGAACGCAAACTTCTGCACTTTCTATGAGAAGCGTATGGCCACAGAAGTTGCTGCTGACGCTTTGGGTGAAGAATGGAAGGGTTATGTGGTCCGAATCAGTGGTGGGAACGACAAACAAGGTTTCCCCATGAAGCAGGGTGTCTTGACCCATGGCCGTGTCCGCCTGCTACTGAGTAAGGGGCATTCCTGTTACAGACCAAGGAGAACTGGAGAAAGAAAGAGAAAATCAGTTCGTGGTTGCATTGTGGATGCAATCTGAGCGTTCTCAACTTGGTTATTGTAAAAAGGAGAGAAGGATATTCCTGGACTGACTGATACTACAGTGCCTCGCCGCCTGGGCCCCAAAAGAGCTAGCAGAATCCGCAAACTTTTCAATCTCTCTAAAAAGAAGATGATGTCCGCCAGTATCGTTGTAAGAAAGCCCTAAAATAAAGAAGGTAAGAAACCTAGGACCAAAGCACCCAAGATTCAGCGTCTGTTACTCCACGTGTCCTGCAGCACAAACGGCGGCGTATTGCTCTGAAGAAGCAGCGTACCAAGAAAAATAAAAGAAGAGGCTGCAGAATATGCTAAACTTTTGGCCTAGAGAATGAAGGAGGCTAAGGAGAAGCGCCAGGAACAAATTGCGAAGAGACGCAGACTTTCCTCTCTGCGGGACTCTACTTCTAAGTCTGAATCCAGTCAGAAATAAGATTTTTTGAGTAACAAATAATAAGATCGGGACTCTGA
// +
// 3A88;IMSJ872377DIJRSRRQRSSRSSSSSSSSGECCKLDIDDGLQRSRRRROSSPRRNOOSSEBCDEJSQKHHJSSSSSSSMMMPSSSSSRJ97677;<SSSSSRRSSSSSSSSSSSSJJKSSSSSSHFFBFGLBCBC<OPLMOP?KSIII6435@ESSSSKSSSSPSSSSD?22275@DB;(((478GIIJKMSIFEFKFA2-)&&''=ALPQQSSRSS,,;>SSSSSSSSSSSSOKGKOSSSSSSSQFLHGISSSSIGGHSSSSFFB.AGA0<AKLM9SSPLLMKMKLJJ..-02::<=0,+-)&&-:?BEFHNLIBA>>E89SSSSSASSQPSOPLMHG7788SSSCB==BCKLMPPPQQKIINRSSSSSSSSSSSSSSSSSPSRPOPPGGCH,,CEH1*%289<DACCRGGHISSSSSSSSQRQRSSSSSMQSRRRFFPPPPPPPPPPPPPPPPP--'.,,/42$))(('')'0314319HFF2104/)*+&#""33%%%(%%$##"""""""&..05%$((*(*36FSSSSSSS8794555HJI0///?SSSSSSSSSSSSSFD110AHKHKKJMJNOPS@;@@@HMQSSLMSOFC>546:<JNSIIIIKJJKSSSSSROO+(((--,1BLJKKSSSSSSSSSSSSSPKJLPSSSSSSSSSMFGPS22116559IIIISQQSSSSSSSSSSSSSQPMB651.13SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSCC>.1ABR+(96;>SSRSQPPPSSSSSSRQMQNQRSSSSSSSSSSMSSSSSMKICDSRQPIF>>>?CL0GHLNMC=<;DBCCBBBCDABBSSSSSRRSSQRMNPPROHGBCFBBBAGGGL@OEC>53038=JSSSROJGKIIHGDDFLOOOSSIBCENLNENFFSSSSNMLF@JSSSPPRQR;:989FLPDBA00//7>PSSSSSSMIHHIRSSJGGIKKPPRRSED;:;?JO=::EKIGBD'&),QQSSSN>S/0KJJIMNORLH@6679FCF5556OOORNNLRQSSPSIII>4HF6A?=AHHSSPOKSLSN-,-?EFMSRSQ;;;DIIGEEIJOSLFE@?*)+-<CD?CFDA999AK;HIFGMQSSSSSSOKCBDSSSOLJ43115AJJGKKLOOMMNLOQSSSSI1,1CCOSS11:IIMSSSSSSSSSSSSOQRRSNJHSSSSSL/../<DEKMLLGPNOA?>>MOSSLKSSSBBCJRRRQSSSSS>76654;<BSONKIKK.--+0,,51+)+450045-,.5OSSHED777SSEEEJSSSSKKIGOOSSSSIIHJSSSSSSLIJOQSSJMSS;EFAA5**)+-2556BBJOM

type Element = i64;
type Matrix = Array2<Element>;
type Tensor = Array3<Element>;

pub const QUAL_OFFSET: u8 = 33;
pub const BASES: &[u8] = b"AUCGN";

#[derive(Debug, Default, Clone)]
pub struct FqEncoderOption {
    kmer_size: u8,
    qual_offset: u8,
    kmer_table: KmerTable,
}

impl FqEncoderOption {
    pub fn new(kmer_size: u8, qual_offset: Option<u8>, base: Option<&[u8]>) -> Self {
        let base = base.unwrap_or(BASES);
        let qual_offset = qual_offset.unwrap_or(QUAL_OFFSET);
        let kmer_table = generate_kmers_table(base, kmer_size);

        Self {
            kmer_size,
            qual_offset,
            kmer_table,
        }
    }
}

#[derive(Debug, Builder)]
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

    fn encoder_seq<'a>(&self, seq: &'a [u8]) -> Vec<&'a [u8]> {
        seq.kmers(self.option.kmer_size).collect()
    }

    pub fn encode_fq(
        &self,
        id: &[u8],
        seq: &[u8],
        qual: &[u8],
        width: usize,
    ) -> Result<(Matrix, Matrix)> {
        // 1.encode the sequence
        // 2.encode the quality

        // normalize to make sure all the bases are consistently capitalized and
        // that we remove the newlines since this is FASTA
        // change unknwon base to 'N'
        let current_width = seq.len() - self.option.kmer_size as usize + 1;

        // all assign -1 as missing value
        let mut encoded_input = Matrix::from_elem((2, width), -1);

        let encoded_seq = self.encoder_seq(seq);
        let encoded_seq_id = encoded_seq
            .par_iter()
            .map(|&s| *self.option.kmer_table.get(s).unwrap())
            .collect::<Vec<usize>>();

        let encoded_qual = self.encode_qual(qual, Some(33));

        // Example encoding logic filling the array based on encoded_seq and encoded_qual
        for i in 0..current_width {
            encoded_input[[0, i]] = encoded_seq_id[i] as Element; // Fill sequence encoding
            encoded_input[[1, i]] = encoded_qual[i] as Element; // Fill quality encoding
        }

        let target = FqEncoder::parse_target_from_id(id).unwrap();
        let mut encoded_target = Matrix::zeros((target.len(), 2));

        for (i, t) in target.iter().enumerate() {
            encoded_target[[i, 0]] = t.start as Element;
            encoded_target[[i, 1]] = t.end as Element;
        }
        Ok((encoded_input, encoded_target))
    }

    // pub fn encoder_fqs<P: AsRef<Path>>(&self, path: P) -> Result<(Tensor, Tensor)> {
    //     let mut reader = parse_fastx_file(path.as_ref()).expect("valid path/file");
    //     let mut records = Vec::new();

    //     let mut max_seq_len = 0;

    //     while let Some(record) = reader.next() {
    //         let seqrec = record.expect("invalid record");
    //         let id = seqrec.id();
    //         let seq = seqrec.normalize(false);
    //         let qual = seqrec.qual().expect("invalid qual");

    //         let seq_len = seqrec.num_bases();
    //         let qual_len = qual.len();
    //         assert_eq!(seq_len, qual_len);

    //         if seq_len > max_seq_len {
    //             max_seq_len = seq_len;
    //         }
    //         records.push((id.to_vec(), seq.to_vec(), qual.to_vec()));
    //     }

    //     let max_width = max_seq_len - self.option.kmer_size as usize + 1;

    //     let records_len = records.len();
    //     let mut concat_input = Array3::<usize>::zeros((records_len, 2, max_width));

    //     let data = records
    //         .par_iter()
    //         .map(|(id, seq, qual)| {
    //             self.encode_fq(id, seq, qual, max_width)
    //                 .expect("encode error")
    //         })
    //         .collect();

    //     // concatenate the encoded input and target
    //     Ok(())
    // }
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
        for kmer in norm_seq.kmers(3) {
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

        assert_eq!(FqEncoder::parse_target_from_id(src).unwrap(), expected);

        // Test case 2: Empty input
        let src = b"";
        let expected: Vec<Interval<usize>> = Vec::new();
        assert_eq!(FqEncoder::parse_target_from_id(src).unwrap(), expected);

        // Test case 3: Invalid input (missing colon)
        let src = b"@462528,100:120";
        assert!(FqEncoder::parse_target_from_id(src).is_err());

        // Test case 4: Invalid input (invalid number)
        let src = b"@462:528,100:abc";
        assert!(FqEncoder::parse_target_from_id(src).is_err());
    }
}
