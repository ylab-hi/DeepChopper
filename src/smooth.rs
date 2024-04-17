use lazy_static::lazy_static;
use rayon::prelude::*;
use std::collections::HashMap;

lazy_static! {
    static ref ID_TABLE: HashMap<i8, char> = {
        let mut m = HashMap::new();
        m.insert(7, 'A');
        m.insert(8, 'C');
        m.insert(9, 'G');
        m.insert(10, 'T');
        m.insert(11, 'N');
        m
    };
}

pub fn id_list2seq(id_list: &[i8]) -> String {
    id_list.par_iter().map(|id| ID_TABLE[&id]).collect()
}

fn majority_voting(labels: &[i32], window_size: usize) -> Vec<i32> {
    // Adjust window size to be odd
    let window_size = if window_size % 2 == 0 {
        window_size + 1
    } else {
        window_size
    };
    let half_window = window_size / 2;
    let len = labels.len();

    labels
        .par_iter()
        .enumerate()
        .map(|(i, _)| {
            let start = i.saturating_sub(half_window);
            let end = usize::min(len, i + half_window + 1);
            let mut window: Vec<i32> = labels[start..end].to_vec();

            // If at the end of the array, extend with the last label to fill the window
            if end == len {
                window.extend(vec![1; i + half_window + 1 - end]); // This assumes '1' is the label to extend
            }

            // Compute the most common element in the window
            let mut counts = HashMap::new();
            for label in window {
                *counts.entry(label).or_insert(0) += 1;
            }
            counts
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(&label, _)| label)
                .unwrap()
        })
        .collect()
}
