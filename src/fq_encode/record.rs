#[derive(Debug)]
pub struct RecordData {
    pub id: Vec<u8>,
    pub seq: Vec<u8>,
    pub qual: Vec<u8>,
}

impl RecordData {
    pub fn new(id: Vec<u8>, seq: Vec<u8>, qual: Vec<u8>) -> Self {
        Self { id, seq, qual }
    }
}

impl From<(Vec<u8>, Vec<u8>, Vec<u8>)> for RecordData {
    fn from(data: (Vec<u8>, Vec<u8>, Vec<u8>)) -> Self {
        Self::new(data.0, data.1, data.2)
    }
}
