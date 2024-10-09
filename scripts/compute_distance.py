import pandas as pd
import sys
import logging
from rich.logging import RichHandler
import numpy as np

from pathlib import Path
import fire
from dataclasses import dataclass


FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

logger = logging.getLogger("__name__")



@dataclass
class Segment:
    chrom: str
    start: int
    end: int

    def overlap(self, other):
        return self.chrom == other.chrom and self.start <= other.end and self.end >= other.start


def load_read_data(filter_full_df):
    from collections import defaultdict
    data = defaultdict(list)

    for r, a in zip(filter_full_df.read_name, filter_full_df.alignments):
        a1, a2 = a.split(",")
        a1_chrom, a1_start, a1_end,_ = a1.split(":")
        a2_chrom, a2_start, a2_end,_ = a2.split(":")

        a1_seg = Segment(a1_chrom, int(a1_start), int(a1_end))
        a2_seg = Segment(a2_chrom, int(a2_start), int(a2_end))

        if a1_seg.chrom > a2_seg.chrom:
            a1_seg, a2_seg = a2_seg, a1_seg

        data[r].append(a1_seg)
        data[r].append(a2_seg)
    return data

def get_similarity(r1, r2, data):
    if r1 == r2:
        return True

    r1_s1, r1_s2 = data[r1]
    r2_s1, r2_s2 = data[r2]

    flag1 = r1_s1.overlap(r2_s1)
    flag2 = r1_s2.overlap(r2_s2)

    return flag1 and flag2


def compute_distance(data, id_to_read):
    data_points = len(data)
    distance_matrix = np.ones((data_points, data_points))
    cache = set()

    for r in range(data_points):
        for c in range(r, data_points):  # Only compute upper triangle

            if (r, c) in cache:
                distance_matrix[r, c] = 0
                distance_matrix[c, r] = 0  # Ensure symmetry

            elif get_similarity(id_to_read[r], id_to_read[c], data):
                cache.add((r, c))
                distance_matrix[r, c] = 0
                distance_matrix[c, r] = 0  # Ensure symmetry

    return distance_matrix


def main(full_df_path: Path, sp_df_path: Path, export_with_sp: bool = False):
    full_df = pd.read_csv(full_df_path, sep='\t', index_col=0)
    sp_df  = pd.read_csv(sp_df_path, sep='\t', index_col=0)

    full_df_reads = set(full_df.index)
    sp_df_reads = set(sp_df.index)
    sups = [ 1 if i in sp_df_reads else 0 for i in full_df_reads]
    full_df['sup'] = sups

    if export_with_sp:
        full_df.to_csv(f'{full_df_path}_with_sp.txt', sep='\t')


    hop_filter = [len(i.split(",")) <= 2   for i in full_df.alignments.values]
    filter_full_df = full_df[hop_filter]
    filter_full_df.reset_index(inplace=True)

    data = load_read_data(filter_full_df)

    id_to_read = {i: r for i, r in enumerate(data.keys())}
    read_to_id = {v:k for k,v in id_to_read.items()}

    logger.info(f"Data points: {len(data)}")
    logger.info("Computing distance matrix...")
    distance_matrix = compute_distance(data, id_to_read)
    logger.info(f"Distance matrix shape: {distance_matrix.shape}")

    np.savez(f"{full_df_path}_distance.npz", distance_matrix=distance_matrix, id_to_read=id_to_read, read_to_id=read_to_id)

if __name__ == "__main__":
    fire.Fire(main)
