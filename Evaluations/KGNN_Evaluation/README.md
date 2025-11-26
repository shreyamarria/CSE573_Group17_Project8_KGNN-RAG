# Phase 1: KGNN Model Evaluation

## Overview
This folder contains the evaluation code and results for the Knowledge Graph Neural Network (KGNN) recommendation model.

## Files

| File | Description |
|------|-------------|
| `evaluation.py` | CPU-based evaluation script |
| `evaluation_gpu_colab.ipynb` | GPU-accelerated Colab notebook |
| `evaluation_metrics_gpu.png` | Visualization of evaluation metrics |
| `evaluation_summary_gpu.csv` | Summary of all metrics |

## Results Summary

**Dataset:** MovieLens 1M (6,040 users, 3,883 movies, 1M ratings)

| Metric | Actual | Target | Status |
|--------|--------|--------|--------|
| **AUC-PR** | 0.834 | ≥0.80 | ✓ Pass |
| **Binary Precision** | 0.904 | ≥0.12 | ✓ Pass |
| AUC-ROC | 0.591 | ≥0.94 | ✗ |
| NDCG@5 | 0.139 | ≥0.35 | ✗ |
| Recall | 0.070 | ≥0.99 | ✗ |
| Precision@5 | 0.144 | ≥0.32 | ✗ |
| F1 Score | 0.129 | ≥0.22 | ✗ |

**Targets Met: 2/9 (22%)**

## Key Insights

- ✅ **High Precision (90.4%)**: When the system predicts "like", it's right 9/10 times
- ✅ **Strong AUC-PR (0.834)**: Effective at ranking liked vs disliked movies
- ⚠️ **Conservative System**: Low recall (7%) means it misses many good recommendations
- ⚠️ **Trade-off**: The model prioritizes accuracy over coverage

## How to Run

1. Upload the notebook to Google Colab
2. Upload the MovieLens CSV files (users.csv, movies.csv, ratings.csv)
3. Select T4 or A100 GPU runtime
4. Run all cells

