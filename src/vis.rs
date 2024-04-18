use colored::Colorize;
use pyo3::prelude::*;
use textwrap;

#[pyfunction]
pub fn highlight_targets(
    sequence: &str,
    targets: Vec<(usize, usize)>,
    text_width: Option<usize>,
) -> String {
    let mut highlighted = String::new();
    let mut last_end = 0;
    for (start, end) in targets {
        highlighted.push_str(&sequence[last_end..start]);
        highlighted.push_str(&sequence[start..end].magenta().bold().to_string());
        last_end = end;
    }
    highlighted.push_str(&sequence[last_end..]);
    let text_width = text_width.unwrap_or(80);
    let chunked = textwrap::wrap(&highlighted, text_width);
    chunked.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highlight() {
        let sequence = "ATGCACTGACTGACATGCACTGACTGAC";

        let hstr = highlight_targets(sequence, vec![(0, 3), (10, 13)], None);

        println!("{}", hstr);
    }
}
