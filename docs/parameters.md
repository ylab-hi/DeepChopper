# Parameter Tuning

**Guidelines for optimizing DeepChopper parameters for your specific dataset.**

Learn how to adjust parameters for different RNA chemistries, dataset characteristics, and use cases.

## Core Parameters

### Sliding Window Size (`--smooth-window`)

The sliding window size controls the smoothing applied to adapter probability scores and significantly impacts detection sensitivity and read fragmentation.
This parameter is applied first in the processing pipeline.

**Recommendation**: Start with the default value of 21. If your analysis reveals excessive fragmentation, increase to 31. If you suspect missed adapters, decrease to 11.

### Minimum Interval Size (`--min-interval-size`)

This parameter determines the minimum length of sequence that can be classified as an adapter region.
This parameter is applied after the smoothing process during adapter region identification.

| Value | Use Case |
| ----- | ------------------------------------------------------------ |
| 8-10 | Increased sensitivity to detect very short adapter fragments |
| 13 | **Default - Balanced detection for typical adapter lengths** |
| 15-20 | Higher precision, reduces false positives in noisy data |

**Recommendation**: For RNA004 chemistry or newer protocols with cleaner data, consider increasing to 15 to reduce false positives.

### Maximum Process Intervals (`--max-process-intervals`)

Limits how many adapter regions are processed per read.
This parameter is applied after the smoothing process.

| Value | Use Case |
| ----- | ------------------------------------------------ |
| 2-3 | Conservative approach for high-quality data |
| 4 | **Default - Suitable for most applications** |
| 5-8 | For highly fragmented reads or complex libraries |

**Recommendation**: Monitor read fragmentation metrics after processing. If reads are being over-fragmented, decrease this value.

### Minimum Read Length (`--min-read-length`)

Specifies the minimum length of sequences to retain after chopping.
This parameter is applied after the smoothing process during the final filtering stage.

| Value | Use Case |
| ----- | ----------------------------------------------- |
| 10 | Small RNA or short fragment applications |
| 20 | **Default - General purpose** |
| 50+ | When only substantial fragments are of interest |

**Recommendation**: Adjust based on your downstream application requirements.

## Chemistry-Specific Recommendations

### RNA002

- Default parameters are optimized based on extensive testing with RNA002 chemistry
- No adjustments needed for typical RNA002 datasets

### RNA004

- Default parameters work well due to DeepChopper's zero-shot capability
- Consider increasing `--min-interval-size` to 15 to account for cleaner data
- May benefit from increasing `--smooth-window` to 31 for reduced false positives

### Newer Chemistries

- Start with RNA004 recommendations
- If performance is suboptimal, first adjust `--smooth-window` and `--min-interval-size`
- Monitor fragmentation metrics to guide further tuning

**Processing Order**: DeepChopper applies parameters in this sequence:

1. Smoothing (`--smooth-window`)
2. Adapter region identification (`--min-interval-size`)
3. Processing adapter regions (`--max-process-intervals`)
4. Final filtering (`--min-read-length`)

## Performance Metrics to Monitor

When tuning parameters, pay attention to these key metrics:

- **Percentage of chimeric alignments** before and after processing
- **Number of segments per read** after processing
- **Read length distribution** after processing
- **Proportion of cDNA-supported alignments** before and after processing

## Need Help?

If you encounter difficulty optimizing DeepChopper for your specific dataset, please open an issue on our GitHub repository with a description of your data characteristics and the results you're observing.
