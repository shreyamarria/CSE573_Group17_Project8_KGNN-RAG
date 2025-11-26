# Evaluations

This folder contains the evaluation code and results for both phases of our KGNN-RAG Movie Recommendation System.

## Structure

```
Evaluations/
├── KGNN_Evaluation/          # Phase 1: Recommendation Model Evaluation
│   ├── evaluation.py
│   ├── evaluation_gpu_colab.ipynb
│   ├── evaluation_metrics_gpu.png
│   ├── evaluation_summary_gpu.csv
│   └── README.md
│
└── RAG_Chatbot_Evaluation/   # Phase 2: Chatbot Interface Evaluation
    ├── CSE573_Chatbot_Evaluation.ipynb
    ├── chatbot_evaluation_metrics.png
    └── README.md
```

## Summary of Results

### Phase 1: KGNN Model (Pre-Chat)
- **Targets Met:** 2/9 (22%)
- **Strengths:** High precision (90.4%), Good AUC-PR (0.834)
- **Weaknesses:** Low recall, conservative recommendations

### Phase 2: RAG Chatbot (Post-Chat)
- **Targets Met:** 4/5 (80%)
- **Strengths:** 100% faithfulness, fast response (145ms), perfect query parsing
- **Weaknesses:** Semantic relevancy below target

## How to Run

Both notebooks are designed to run on Google Colab with GPU acceleration.

1. Upload the notebook to Colab
2. Upload MovieLens data files (users.csv, movies.csv, ratings.csv)
3. Select GPU runtime (T4 recommended)
4. Run all cells

