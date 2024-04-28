import json
import pysam
import deepchopper
from deepchopper import (
    remove_intervals_and_keep_left,
)
from deepchopper.utils import highlight_targets
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import logging
from rich.logging import RichHandler
from rich.progress import track
from needletail import (
    parse_fastx_file,
    NeedletailError,
)

import seaborn as sns
import re
from textwrap import wrap
import gget


FORMAT = "%(message)s"
log_level = logging.INFO
logging.basicConfig(
    level=log_level,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)
log = logging.getLogger("deepchopper")
log.setLevel(log_level)


INTERNAL_THRESHOLD: float = 0.9
OVERLAP_THRESHOLD: float = 0.4
BLAT_THRESHOLD: float = 0.9
MIN_MAPPING_QUALITY: int = 0

SMOOTH_WINDOW_SIZE: int = 21
MIN_INTERVAL_SIZE: int = 10
APPROVED_INTERVAL_NUMBER: int = 10
PLOYA_THRESHOLD: int = 3


@dataclass
class FqRecord:
    id: str
    seq: str
    qual: str

    def to_str(self):
        return f"{self.id}\n{self.seq}\n+\n{self.qual}"


def vis_qual_static(predict, start: int | None = None, end: int | None = None, figure_size=(20, 1)):
    if predict.qual is None:
        raise ValueError("no qual, please fetch qual first")

    start = 0 if start is None else start
    end = len(predict.seq) if end is None else end

    qual = np.array([ord(c) - 33 for c in list(predict.qual[start:end])]).reshape(1, -1)
    seq = list(predict.seq[start:end])

    # Creating the heatmap
    fig, ax = plt.subplots(figsize=figure_size)  # Set a wide figure to accommodate the sequence
    cax = ax.imshow(qual, aspect="auto", cmap="viridis")
    cbar = plt.colorbar(cax, ax=ax, orientation="vertical")
    cbar.set_label("Value")
    # Setting up the sequence as x-axis labels
    ax.set_xticks(np.arange(len(seq)))
    ax.set_xticklabels(seq, rotation=90)  # Rotate labels for better readability
    # Remove y-axis labels as there's only one row
    ax.set_yticks([])
    ax.set_title(f"{predict.id}: {start}-{end}")
    plt.show()
    plt.close()


def to_fqs_record(predict, intervals: list[tuple[int, int]]):
    if predict.qual is None:
        raise ValueError("no qual, please fetch qual first")

    assert len(predict.qual) == len(predict.seq)

    seqs, saved_intervals = remove_intervals_and_keep_left(predict.seq, intervals)
    quals, saved_intervals = remove_intervals_and_keep_left(predict.qual, intervals)

    assert len(seqs) == len(quals)
    for ind, (seq, qual) in enumerate(zip(seqs, quals, strict=True)):
        record_id = f"@{predict.id}|{saved_intervals[ind][0], saved_intervals[ind][1]}"
        yield FqRecord(id=record_id, seq=seq, qual=qual)


def smooth_and_select_intervals(
    predict_id,
    stats,
    smooth_window_size: int,
    min_interval_length: int,
    approved_interval_nums: int = 1,
) -> list[tuple[int, int]]:
    chop_intervals = stats.smooth_intervals[predict_id]

    results = []
    for interval in chop_intervals:
        if interval[1] - interval[0] > min_interval_length:
            results.append(interval)

    if len(results) > approved_interval_nums:
        return []

    return results


def collect_fq_records(file: Path):
    result = {}
    try:
        for record in parse_fastx_file(file.as_posix()):
            result[record.id] = record
    except NeedletailError:
        log.error("Invalid Fastq file")

    return result


def collect_sam_records(file: Path):
    if not isinstance(file, Path):
        file = Path(file)

    result = {}
    samfile = pysam.AlignmentFile(file.as_posix(), "rb")

    for read in samfile.fetch():
        result[read.query_name] = read

    return result


def vis_hist_for_num_of_intervals(data, figsize=(10, 6), title=None, ax=None, set_xticks=False):
    # Create histogram with a kernel density estimate
    max_x = max(data) + 1
    if ax is None:
        plt.figure(figsize=figsize)
        sns.histplot(data, kde=True, color="#66c2a5", line_kws={"linewidth": 2}, discrete=True)
        if set_xticks:
            plt.xticks(range(0, max_x, 1))
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
    else:
        sns.histplot(
            data,
            kde=True,
            color="#66c2a5",
            line_kws={"linewidth": 2},
            discrete=True,
            ax=ax,
        )
        ax.set_title(title)
        if set_xticks:
            ax.set_xticks(range(0, max_x, 1))
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")


def wrap_str(ostr, width):
    return "\n".join(wrap(ostr, width))


def show_sam_record(predict, stats, sam_records):
    seq_len = len(predict.seq)
    txt_width = 120

    log.debug(f"read id {predict.id} seq len: {seq_len}")

    smooth_intervals = stats.smooth_intervals[predict.id]

    for interval in smooth_intervals:
        quals = predict.qual_array()[interval[0] : interval[1]]
        average_qual = sum(quals) / len(quals)
        log.debug(f"smooth interval : {interval} len: {interval[1] - interval[0]}     {average_qual=}")

    highlight_targets(predict.seq, predict.prediction_region())
    highlight_targets(predict.seq, smooth_intervals)

    predict_read = sam_records.get(predict.id, None)
    if predict_read is None:
        log.debug("the read is not map")
        return

    if len(smooth_intervals) == 1:
        blat_df = gget.blat(predict.seq[smooth_intervals[0][0] : smooth_intervals[0][1]])
        if blat_df is not None:
            log.debug(f"blat result:\n {blat_df.to_string()}\n")

    log.debug(
        f"{predict_read.reference_id=} strand={'+' if predict_read.is_forward else '-'} {predict_read.mapping_quality=}"
    )
    log.debug(f"{predict_read.reference_start=} {predict_read.reference_end=}")
    log.debug(f"cigar: {wrap_str(predict_read.cigarstring, txt_width)}")

    ls_len, rs_len = deepchopper.left_right_soft_clip(predict_read.cigarstring)
    if not predict_read.is_forward:
        ls_len, rs_len = rs_len, ls_len

    log.debug(f"ls: 0-{ls_len}  \n {wrap_str(predict.seq[:ls_len], txt_width)}")
    log.debug(f"rs: {seq_len-rs_len}-{seq_len} \n {wrap_str(predict.seq[-rs_len:], txt_width)}")

    if predict_read.has_tag("SA"):
        log.debug(f"SA: {predict_read.get_tag('SA')}")
        chimeric_alns = predict_read.get_tag("SA")[:-1].split(";")

        for _aln in chimeric_alns:
            (
                chr_sa,
                pos_sa,
                strand_sa,
                cigar_sa,
                mapq_sa,
                nm_sa,
            ) = _aln.split(",")

            left_mat = pat_left_s.search(cigar_sa)
            right_mat = pat_right_s.search(cigar_sa)

            l_s_len = left_mat.group(1) if left_mat else ""
            r_s_len = right_mat.group(1) if right_mat else ""

            tgt_key = f"{predict_read.qname}\t{l_s_len=}\t{r_s_len=}"
            log.debug(f"chimeric : {tgt_key}")


def check_overlap(interval1: tuple[int, int], interval2: tuple[int, int], overlap_threshold: float) -> bool:
    # interval2 is predicted region

    start1, end1 = interval1
    start2, end2 = interval2

    length1 = end1 - start1
    length2 = end2 - start2

    # Calculate the maximum start point and minimum end point
    max_start = max(start1, start2)
    min_end = min(end1, end2)

    # union
    min_start = min(start1, start2)
    max_end = max(end1, end2)

    # Calculate the overlap length
    overlap = max(0, min_end - max_start)

    divide = length2

    ratio = overlap / divide

    # Check if the overlap meets or exceeds the threshold
    log.debug(f"compare {interval1}({length1}) {interval2}({length2}) {ratio=}")
    return ratio >= overlap_threshold


def process_one_interval_parallel(
    overlap_results,
    whole_seq_len: int,
    pseq,
    pid,
    ls_len: int,
    rs_len: int,
    pd_start: int,
    pd_end: int,
    overlap_threshold: float,
    internal_threshold: float,
    blat_threshold: float,
    read_mp: int,
    min_mapping_quality: int,
):
    predict_seq = pseq[pd_start:pd_end]
    min_blat_seq_len = 20

    if pd_end / whole_seq_len > internal_threshold:
        # terminal adapter
        # has overlap
        if check_overlap(
            (whole_seq_len - rs_len, whole_seq_len),
            (pd_start, pd_end),
            overlap_threshold,
        ):
            overlap_results["terminal_chop_sc"].append(pid)
        else:
            overlap_results["terminal_chop_nosc"].append(pid)
            if len(predict_seq) < min_blat_seq_len:
                overlap_results["terminal_chop_nosc_cannot_blat"].append(pid)
                return

            blat_df = gget.blat(predict_seq)
            if blat_df is not None:
                log.debug(f"blat_df: {blat_df.to_string()}\n")
            else:
                log.debug("blat_df is None")

            if blat_df is None or (blat_df.iloc[0]["%_aligned"] / 100 < blat_threshold):
                overlap_results["terminal_chop_nosc_noblat"].append(pid)

    else:  # internal adapter
        flag = False
        if ls_len != 0:
            if check_overlap((0, ls_len), (pd_start, pd_end), overlap_threshold):
                flag = True
                overlap_results["internal_chop_sc"].append(pid)

        if rs_len != 0 and not flag:
            if check_overlap(
                (whole_seq_len - rs_len, whole_seq_len),
                (pd_start, pd_end),
                overlap_threshold,
            ):
                flag = True
                overlap_results["internal_chop_sc"].append(pid)

        if not flag:
            overlap_results["internal_chop_nosc"].append(pid)

            if len(predict_seq) < min_blat_seq_len:
                # seq is too short, and cannot use blat
                overlap_results["internal_chop_nosc_cannot_blat"].append(pid)
                return

            blat_df = gget.blat(predict_seq)
            if blat_df is not None:
                log.debug(f"blat_df: {blat_df.to_string()}\n")
            else:
                log.debug("blat_df is None")

            if blat_df is None or (blat_df.iloc[0]["%_aligned"] / 100 < blat_threshold):
                overlap_results["internal_chop_nosc_noblat"].append(pid)


def verify_result_with_sam_records_rs(
    overlap_results,
    predict,
    stats,
    rs_read,
    internal_threshold: float = INTERNAL_THRESHOLD,
    overlap_threshold: float = OVERLAP_THRESHOLD,
    blat_threshold: float = BLAT_THRESHOLD,
    min_mapping_quality: int = MIN_MAPPING_QUALITY,
):
    read_mapping_quality = rs_read.mapping_quality

    if not rs_read.is_mapped:
        log.debug(f"the read {predict.id} is not map")
        overlap_results["unmap_read"].append(predict.id)
        return

    if read_mapping_quality < min_mapping_quality:
        log.debug(f"the read {predict.id}'s mapping_quality {read_mapping_quality} is low")
        overlap_results["low_mp_read"].append(predict.id)
        return

    seq_len = len(predict.seq)
    ls_len, rs_len = rs_read.left_softclip, rs_read.right_softclip

    intervals = stats.smooth_intervals[predict.id]

    log.debug(predict.show_info(intervals))
    txt_width = 120
    log.debug(f"strand={'+' if rs_read.is_forward else '-'} {rs_read.mapping_quality=}")
    log.debug(f"cigar: {wrap_str(rs_read.cigar, txt_width)}")
    log.debug(f"ls {ls_len}: 0-{ls_len}  \n {wrap_str(predict.seq[:ls_len], txt_width)}")
    log.debug(f"rs {rs_len}: {seq_len-rs_len}-{seq_len} \n {wrap_str(predict.seq[seq_len-rs_len:seq_len], txt_width)}")

    if len(intervals) == 1:
        # clean predict
        start, end = intervals[0]
        # quals = predict.qual_array()[start:end]
        # average_qual = sum(quals) / len(quals)
        process_one_interval_parallel(
            overlap_results,
            seq_len,
            predict.seq,
            predict.id,
            ls_len,
            rs_len,
            start,
            end,
            overlap_threshold,
            internal_threshold,
            blat_threshold,
            read_mapping_quality,
            min_mapping_quality,
        )
    elif len(intervals) <= 3:
        for interval in intervals:
            start, end = interval
            process_one_interval_parallel(
                overlap_results,
                seq_len,
                predict.seq,
                predict.id,
                ls_len,
                rs_len,
                start,
                end,
                overlap_threshold,
                internal_threshold,
                blat_threshold,
                read_mapping_quality,
                min_mapping_quality,
            )
    else:
        overlap_results["no_process"].append(predict.id)
        pass


def verify_result_with_sam_records_for_parallel(
    pseq: str,
    pid: str,
    smooth_intervals: dict[str, list[tuple[int, int]]],
    read_is_mapped: bool,
    read_mapping_quality: int,
    read_left_softclip: int,
    read_right_softclip: int,
    internal_threshold: float = INTERNAL_THRESHOLD,
    overlap_threshold: float = OVERLAP_THRESHOLD,
    blat_threshold: float = BLAT_THRESHOLD,
    min_mapping_quality: int = MIN_MAPPING_QUALITY,
):
    overlap_results = defaultdict(list)
    read_mapping_quality = read_mapping_quality

    if not read_is_mapped:
        log.debug(f"the read {pid} is not map")
        overlap_results["unmap_read"].append(pid)
        return overlap_results

    if read_mapping_quality < min_mapping_quality:
        log.debug(f"the read {pid}'s mapping_quality {read_mapping_quality} is low")
        overlap_results["low_mp_read"].append(pid)
        return overlap_results

    seq_len = len(pseq)
    ls_len, rs_len = read_left_softclip, read_right_softclip
    intervals = smooth_intervals[pid]

    if len(intervals) == 1:
        # clean predict
        start, end = intervals[0]

        process_one_interval_parallel(
            overlap_results,
            seq_len,
            pseq,
            pid,
            ls_len,
            rs_len,
            start,
            end,
            overlap_threshold,
            internal_threshold,
            blat_threshold,
            read_mapping_quality,
            min_mapping_quality,
        )
    elif len(intervals) <= 3:
        for interval in intervals:
            start, end = interval
            process_one_interval_parallel(
                overlap_results,
                seq_len,
                pseq,
                pid,
                ls_len,
                rs_len,
                start,
                end,
                overlap_threshold,
                internal_threshold,
                blat_threshold,
                read_mapping_quality,
                min_mapping_quality,
            )
    else:
        overlap_results["no_process"].append(pid)

    return overlap_results


def merge_results(results_list):
    combined_results = defaultdict(list)
    for single_result in results_list:
        for key, values in single_result.items():
            combined_results[key].extend(values)
    return combined_results


def get_acc(data):
    internal_chop_sc_count = len(data.get("internal_chop_sc", []))
    internal_chop_nosc_count = len(data.get("internal_chop_nosc", []))
    internal_chop_nosc_noblat_count = len(data.get("internal_chop_nosc_noblat", []))
    internal_chop_nosc_cannotblat_count = len(data.get("internal_chop_nosc_cannot_blat", []))
    total_internal = internal_chop_sc_count + internal_chop_nosc_count
    confirmed_internal = internal_chop_sc_count + internal_chop_nosc_noblat_count

    internal_acc = confirmed_internal / (total_internal)

    terminal_chop_sc_count = len(data.get("terminal_chop_sc", []))
    terminal_chop_nosc_count = len(data.get("terminal_chop_nosc", []))
    terminal_chop_nosc_noblat_count = len(data.get("terminal_chop_nosc_noblat", []))
    terminal_chop_nosc_cannotblat_count = len(data.get("terminal_chop_nosc_cannot_blat", []))
    total_terminal = terminal_chop_sc_count + terminal_chop_nosc_count
    confirmed_terminal = terminal_chop_sc_count + terminal_chop_nosc_noblat_count

    terminal_acc = confirmed_terminal / (total_terminal)

    total_acc = (confirmed_internal + confirmed_terminal) / (total_internal + total_terminal)

    return internal_acc, terminal_acc, total_acc


def vis_overlap_results(data):
    import pandas as pd

    internal_acc, terminal_acc, total_acc = get_acc(data)

    plot_df = pd.DataFrame(
        [(key, len(value)) for key, value in data.items()],
        columns=["Category", "Count"],
    )

    # Plotting the data
    plt.figure(figsize=(10, 6))  # Set the figure size
    bars = plt.bar(plot_df["Category"], plot_df["Count"], color="skyblue")  # Create a bar chart
    # Add text annotations to the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.5,
            yval,
            ha="center",
            va="bottom",
        )

    plt.xlabel("Category")  # Set the x-label
    plt.ylabel("Number of Items")  # Set the y-label
    plt.title(
        f"Count of Items in Each Category {internal_acc=:.4f} {terminal_acc=:.4f} {total_acc=:.4f}"
    )  # Set the title
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better visibility
    plt.tight_layout()  # Adjust layout to make room for the rotated x-labels
    # plt.show()  # Display the plot
    plt.savefig("overlap_results.pdf", dpi=300)
    plt.close()

def vis_stats(stats, total:int):
    # Extracting data for plotting
    categories = ['Total Predicts', 'Total Truncated', 'Predicts with Chop', 'Smooth Predicts with Chop', 'Smooth Internal Predicts',
                   'Smooth Only One', 'Smooth Polya Only One']
    values = [
        stats.total_predicts,
        stats.total_truncated,
        len(stats.predicts_with_chop),
        len(stats.smooth_predicts_with_chop),
        len(stats.smooth_internal_predicts),
        len(stats.smooth_only_one),
        len(stats.smooth_only_one_with_ploya),
    ]

    # Creating the bar plot
    plt.figure(figsize=(10, 6))
    bars  = plt.bar(categories, values,color="#66c2a5")

    # Add text annotations to the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.5,
            yval,
            ha="center",
            va="bottom",
        )

    plt.xlabel('Categories')
    plt.ylabel('Count')
    plt.title(f'Statistics for {total}')
    plt.xticks(rotation=45, ha="right")  # Rotate category names for better visibility
    plt.tight_layout()  # Adjust layout to make all labels visible
    # plt.show()
    plt.savefig("stats.pdf", dpi=300)


def main():
    bam_file = "/projects/b1171/ylk4626/project/DeepChopper/data/eval/real_data/dorado_without_trim_fqs/VCaP.bam"
    rs_sam_records = deepchopper.read_bam_records_parallel(bam_file)
    log.debug(f"total sam records: {len(rs_sam_records)}")

    ## VCaP
    hyena_results = [
        Path("/projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap/VCaP.fastq_0/predicts/0/"),
        Path("/projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap/VCaP.fastq_1/predicts/0/"),
        Path("/projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap/VCaP.fastq_2/predicts/0/"),
        Path("/projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap/VCaP.fastq_3/predicts/0/"),
        Path("/projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap/VCaP.fastq_4/predicts/0/"),
        Path("/projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap/VCaP.fastq_5/predicts/0/"),
        Path("/projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap/VCaP.fastq_6/predicts/0/"),
        Path("/projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap/VCaP.fastq_7/predicts/0/"),
        Path("/projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap/VCaP.fastq_8/predicts/0/"),
        Path("/projects/b1171/ylk4626/project/DeepChopper/logs/eval/runs/vcap/VCaP.fastq_9/predicts/0/"),
    ]

    max_batches = 1000
    all_predicts = deepchopper.load_predicts_from_batch_pts(hyena_results[0], -100, max_batches)

    stats = deepchopper.py_collect_statistics_for_predicts_parallel(
        list(all_predicts.values()),
        smooth_window_size=SMOOTH_WINDOW_SIZE,
        min_interval_size=MIN_INTERVAL_SIZE,
        approved_interval_number=APPROVED_INTERVAL_NUMBER,
        internal_threshold=INTERNAL_THRESHOLD,
        ploya_threshold=PLOYA_THRESHOLD,
    )
    vis_stats(stats, len(all_predicts))

    original_prediction_number = stats.number_predicts_with_chop(all_predicts)
    smooth_prediction_number = stats.number_smooth_predicts_with_chop()
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    vis_hist_for_num_of_intervals(original_prediction_number, title="Original Intervals", ax=axs[0])
    vis_hist_for_num_of_intervals(smooth_prediction_number, title="Smooth Intervals", ax=axs[1])
    fig.savefig("predicts_intervals_number.pdf", dpi=300)

    plot_oregion_size_data = stats.length_predicts_with_chop(all_predicts)
    plot_sregion_size_data = stats.lenghth_smooth_predicts_with_chop()
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    vis_hist_for_num_of_intervals(
        plot_oregion_size_data,
        title=f"Chop Size of clean data (original) {min(plot_oregion_size_data)}-{max(plot_oregion_size_data)}",
        ax=axs[0],
    )
    vis_hist_for_num_of_intervals(
        plot_sregion_size_data,
        title=f"Chop Size of clean data (smooth) {min(plot_sregion_size_data)}-{max(plot_sregion_size_data)}",
        ax=axs[1],
    )
    fig.savefig("predicts_intervals_size.pdf", dpi=300)

    total_predicts = len(stats.smooth_predicts_with_chop)
    overlap_results = defaultdict(list)
    for p in track(stats.smooth_predicts_with_chop, description=f"Processing {total_predicts} predicts..."):
        verify_result_with_sam_records_rs(overlap_results, all_predicts[p], stats, rs_sam_records[p])

    with open(f"overlap_result_{max_batches}.json", "w") as outfile:
        json.dump(overlap_results, outfile, indent=4, sort_keys=False)

    vis_overlap_results(overlap_results)

def vis():
    pass

if __name__ == "__main__":
    main()
