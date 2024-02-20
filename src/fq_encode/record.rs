use bstr::BString;

#[derive(Debug, Default)]
pub struct RecordData {
    pub id: BString,
    pub seq: BString,
    pub qual: BString,
}

impl RecordData {
    pub fn new(id: BString, seq: BString, qual: BString) -> Self {
        Self { id, seq, qual }
    }
}

impl From<(Vec<u8>, Vec<u8>, Vec<u8>)> for RecordData {
    fn from(data: (Vec<u8>, Vec<u8>, Vec<u8>)) -> Self {
        Self::new(data.0.into(), data.1.into(), data.2.into())
    }
}

impl From<(BString, BString, BString)> for RecordData {
    fn from(data: (BString, BString, BString)) -> Self {
        Self::new(data.0, data.1, data.2)
    }
}
