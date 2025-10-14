#!/usr/bin/env Rscript
library(limma)

# Parse arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Usage: Rscript run_limma.R <expression.csv> <design.csv> <contrast> <output.csv>")
}

expr_file <- args[1]
design_file <- args[2]
contrast_string <- args[3]
output_file <- args[4]

# Parse contrast string to get limma format
contrast_parts <- strsplit(contrast_string, "_VS_")[[1]]
if (length(contrast_parts) != 2) {
  stop(sprintf("Invalid contrast format: %s. Expected 'level1_VS_level2'", contrast_string))
}
level1 <- contrast_parts[1]
level2 <- contrast_parts[2]

# Read data
cat("Reading expression data...\n")
expr_data <- read.csv(expr_file, row.names=1, check.names=FALSE)
expr_matrix <- as.matrix(expr_data)

cat("Reading design matrix...\n")
design_info <- read.csv(design_file, row.names=1, stringsAsFactors=FALSE)

# Ensure condition is a factor with levels in correct order
design_info$condition <- factor(design_info$condition, levels=c(level1, level2))

# Create design matrix without intercept
design_matrix <- model.matrix(~ 0 + condition, data=design_info)

# set column names back to actual levels
colnames(design_matrix) <- levels(design_info$condition)

cat(sprintf("Design matrix dimensions: %d samples x %d groups\n",
            nrow(design_matrix), ncol(design_matrix)))
cat(sprintf("Expression matrix dimensions: %d proteins x %d samples\n",
            nrow(expr_matrix), ncol(expr_matrix)))

# Check that sample names match
if (!all(colnames(expr_matrix) %in% rownames(design_matrix))) {
  stop("Sample names in expression matrix don't match design matrix")
}

# Reorder expression matrix columns to match design matrix rows --> ensure matching during fit
expr_matrix <- expr_matrix[, rownames(design_matrix)]

# Run limma
cat("Fitting linear model...\n")
fit <- lmFit(expr_matrix, design_matrix)

# Make contrast: level2 - level1 (treatment - control typically)
contrast_formula <- sprintf("%s-%s", level2, level1)
cat(sprintf("Computing contrast: %s\n", contrast_formula))

contrast_matrix <- makeContrasts(contrasts=contrast_formula, levels=design_matrix)
fit2 <- contrasts.fit(fit, contrast_matrix)
fit2 <- eBayes(fit2)

# Get all results, keeping original order
results <- topTable(fit2, number=Inf, sort.by="none")

# Add protein names as explicit column for clarity
results$protein <- rownames(results)
results <- results[, c("protein", "logFC", "AveExpr", "t", "P.Value", "adj.P.Val", "B")]

# Write output
cat(sprintf("Writing results to %s\n", output_file))
write.csv(results, output_file, row.names=FALSE)

# Summary statistics
n_sig_005 <- sum(results$adj.P.Val < 0.05, na.rm=TRUE)
n_sig_01 <- sum(results$adj.P.Val < 0.1, na.rm=TRUE)
cat(sprintf("Significant proteins (FDR < 0.05): %d\n", n_sig_005))
cat(sprintf("Significant proteins (FDR < 0.10): %d\n", n_sig_01))
cat("Done!\n")
