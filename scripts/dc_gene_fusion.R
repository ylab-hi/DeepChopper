#!/usr/bin/env Rscript

# Load required libraries
suppressPackageStartupMessages({
  library(optparse)
  library(chimeraviz)
  library(svglite)
  library(systemfonts)
})

# Define command line options
option_list <- list(
  make_option(c("-i", "--input"), type="character", default=NULL,
              help="Input file path for Infusion data", metavar="FILE"),
  make_option(c("-o", "--output"), type="character", default="fusion_plot.svg",
              help="Output SVG file name [default= %default]", metavar="FILE"),
  make_option(c("-w", "--width"), type="numeric", default=10,
              help="Plot width in inches [default= %default]", metavar="NUMBER"),
  make_option(c("--height"), type="numeric", default=10,
              help="Plot height in inches [default= %default]", metavar="NUMBER"),
  make_option(c("--help"), action="store_true", default=FALSE,
              help="Show this help message and exit")
)

# Create option parser object with add_help_option=FALSE
opt_parser <- OptionParser(option_list=option_list,
                           usage = "usage: %prog [options]",
                           description = "\nThis script generates a fusion circle plot using Infusion data and saves it as an SVG file.",
                           epilogue = "\nExample usage:\n  Rscript %prog -i input_data.txt -o my_fusion_plot.svg -w 12 --height 12",
                           add_help_option=FALSE)

# Parse command line arguments
opt <- parse_args(opt_parser)

# Debug: Print parsed options
cat("Parsed options:\n")
print(opt)

# Check if help is requested
if (opt$help) {
  print_help(opt_parser)
  quit(status = 0)
}

# Check if input file is provided
if (is.null(opt$input)) {
  cat("Error: Input file must be specified. Use -i or --input option.\n\n")
  print_help(opt_parser)
  quit(status = 1)
}

# Debug: Print input file path
cat("Input file:", opt$input, "\n")

# Load Infusion data
tryCatch({
  fusions <- import_infusion(opt$input, "hg38")
  cat("Infusion data loaded successfully.\n")
}, error = function(e) {
  cat("Error loading Infusion data:", conditionMessage(e), "\n")
  quit(status = 1)
})

tryCatch({
  svglite(opt$output, width = opt$width, height = opt$height, system_fonts = list(sans = "Arial"))
  plot_circle(fusions)
  dev.off()
  cat(paste("SVG plot saved as", opt$output, "\n"))
}, error = function(e) {
  cat("Error creating SVG:", conditionMessage(e), "\n")
  quit(status = 1)
})

cat("Script execution completed.\n")
