# CNLA
This repository contains the full implementation and related components of CNLA (Classwise Normalization with Learnable Alignment) — a simple yet effective preprocessing framework for improving robustness under distributional shift.

📁 Project Structure
CNLA/
Core implementation of the CNLA module and utility functions.

setup/
Configuration and setup scripts. Includes dataset loaders and training utilities.

test/
Evaluation scripts for testing the performance of CNLA on various datasets. Compares accuracy and F1 scores with and without CNLA under artificially shifted distributions.

baseline/
Other normalization strategies used for comparison, such as:

StandardScaler (global normalization)

Classwise Z-score normalization

BatchNorm-style transformation

CORAL (covariance alignment)

🔍 Description
CNLA integrates classwise normalization with a lightweight learnable alignment module. It is particularly useful for classification tasks involving tabular or structured data, where shifts in data distribution can degrade model performance. This repository provides:

A plug-and-play PyTorch implementation

Support for pseudo-label-based test-time adaptation

Flexible benchmarking across multiple datasets

📊 Benchmarking
All experiments follow a 5-fold stratified cross-validation protocol. Metrics reported include:

Accuracy

F1 Score (Macro & Micro)

Performance is compared before and after applying CNLA, as well as against standard baseline normalization techniques.

📫 Contact
For questions or collaborations, feel free to reach out or open an issue.
