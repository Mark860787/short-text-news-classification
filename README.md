# Short-Text News Classification under Reduced Context

**COMP90051 Statistical Machine Learning — University of Melbourne**
**Course Project | Final Score: 29.25 / 30**

This project investigates how **textual feature construction under reduced context affects classifier generalization in short-text news classification**.

I compare models with different levels of complexity — **Logistic Regression, BiLSTM, and fine-tuned DeBERTa** — across progressively richer textual context levels to understand how much classification performance depends on the amount and quality of information available to a model.

📄 **[View the full report](https://github.com/Mark860787/short-text-news-classification/blob/main/Report.pdf)**

---

## Project Motivation

As digital information continues to grow, people increasingly need to obtain useful information from shorter pieces of text. Headlines, notifications, search results, previews, and summaries are all designed to communicate as much information as possible while requiring minimal reading time.

This motivated us to ask a practical machine learning question:

> **How little textual context is enough for a model to still understand and classify the underlying topic?**

Rather than only asking which model achieves the highest classification accuracy, this project studies how model performance changes as textual information is progressively reduced.

We are particularly interested in whether:

* richer feature representations can compensate for missing context;
* more complex architectures remain advantageous when only a few words are available; and
* there is a point at which adding more text provides only limited additional benefit.

This makes short-text classification not only a model-comparison problem, but also an investigation into the trade-off between **information availability, representation quality, and model complexity**.

---

## Research Question

The central research question is:

> **How does textual feature construction under reduced context affect the generalization of classifiers with different levels of model complexity in short-text news topic classification?**

The experiments progressively expand the input from a highly truncated headline to a full headline and short description, allowing us to directly measure the value of additional textual context.

---

## Dataset

The project uses the **News Category Dataset** by Rishabh Misra.

The experiments focus on the top **10 news categories**, with **2,000 samples per category**, producing a balanced dataset of:

**20,000 news articles**

The dataset is not included in this repository.

Dataset source:

https://www.kaggle.com/datasets/rmisra/news-category-dataset

The expected original dataset file is:

```text
News_Category_Dataset.json
```

---

## Context Construction

To systematically study information reduction, the input text is divided into four progressively richer context levels:

| Context Level | Input                                              |
| ------------- | -------------------------------------------------- |
| **L1**        | First 5 words of the headline                      |
| **L2**        | Full headline                                      |
| **L3**        | Headline + first 15 words of the short description |
| **L4**        | Headline + full short description                  |

This design allows the effect of textual context to be studied independently from model architecture.

---

## Models and Representations

The project compares three levels of model complexity.

### 1. Logistic Regression

Logistic Regression is used as the simple linear baseline.

Three feature representations are evaluated:

* **TF-IDF**
* **TF-IDF + POS features**
* **Frozen BERT sentence embeddings**

For TF-IDF, word-level and character-level n-gram features are combined. POS-based features are added as additional syntactic information.

For the BERT variant, `bert-base-uncased` is used as a frozen feature extractor and its token embeddings are mean-pooled into sentence representations.

---

### 2. BiLSTM

BiLSTM represents the intermediate-complexity model.

Two embedding strategies are compared:

* **BiLSTM + Word2Vec**
* **BiLSTM + frozen BERT token embeddings**

The Word2Vec variant uses 100-dimensional skip-gram embeddings trained only on the corresponding training data.

The frozen-BERT variant uses contextual token-level representations from `bert-base-uncased`, allowing the contribution of stronger textual representation to be evaluated while keeping the BiLSTM architecture fixed.

---

### 3. Fine-tuned DeBERTa

The most complex model is:

```text
microsoft/deberta-base
```

DeBERTa is fine-tuned end-to-end for the **10-class news classification task**.

Unlike the frozen-BERT approaches, DeBERTa adapts its contextual representations directly to the classification objective.

---

## Experimental Design

All model families are evaluated using the same **nested stratified cross-validation framework**.

### Outer evaluation

* **10-fold stratified cross-validation**
* Same outer folds for all model families
* Each outer test fold contains approximately 2,000 articles

### Inner hyperparameter selection

* **3-fold stratified cross-validation**
* Hyperparameters are selected using mean **Macro F1**
* The selected configuration is retrained on the complete outer-training partition
* Evaluation is then performed once on the held-out outer-test fold

Using identical outer folds ensures that comparisons between model families are paired rather than based only on similarly distributed test sets.

---

## Evaluation Metrics

The models are evaluated using:

* **Accuracy**
* **Macro F1**
* **Weighted F1**
* **Per-class F1**

Because the dataset is balanced across the 10 categories, **Macro F1** is used as the primary model-comparison metric.

---

## Key Results

| Model                              | L1 Macro F1 | L2 Macro F1 | L3 Macro F1 | L4 Macro F1 |
| ---------------------------------- | ----------: | ----------: | ----------: | ----------: |
| Logistic Regression + TF-IDF       |       0.562 |       0.690 |       0.716 |       0.735 |
| Logistic Regression + TF-IDF + POS |       0.564 |       0.691 |       0.721 |       0.739 |
| Logistic Regression + Frozen BERT  |       0.570 |       0.702 |       0.732 |       0.750 |
| BiLSTM + Word2Vec                  |       0.188 |       0.325 |       0.435 |       0.489 |
| BiLSTM + Frozen BERT               |       0.558 |       0.688 |       0.721 |       0.733 |
| **Fine-tuned DeBERTa**             |   **0.640** |   **0.767** |   **0.812** |   **0.835** |

---

## Main Findings

### 1. More context consistently improves classification

All major model configurations improve from L1 to L4.

For example, DeBERTa Macro F1 increases from:

```text
0.640 → 0.767 → 0.812 → 0.835
```

This confirms that missing textual evidence remains a major limitation in short-text classification.

---

### 2. Fine-tuned DeBERTa performs best at every context level

Fine-tuned DeBERTa consistently achieves the highest Macro F1.

Its advantage over Logistic Regression with frozen BERT also increases as more textual context becomes available, suggesting that end-to-end contextual adaptation becomes increasingly useful when richer semantic information is present.

---

### 3. Representation quality can matter more than architecture complexity

A more complicated model does not automatically perform better.

BiLSTM with Word2Vec performs substantially worse than the simpler Logistic Regression + TF-IDF baseline.

However, replacing Word2Vec with frozen BERT token representations allows the same BiLSTM architecture to recover performance close to Logistic Regression + BERT.

This suggests that for very short text:

> **the quality of the textual representation can be more important than simply increasing model complexity.**

---

### 4. Additional context shows diminishing returns

Although performance increases monotonically from L1 to L4, the improvement becomes progressively smaller.

For DeBERTa:

```text
L1 → L2: +0.127
L2 → L3: +0.045
L3 → L4: +0.023
```

The relatively small L3 → L4 improvement suggests that much of the useful semantic information is already contained in the headline and the beginning of the short description.

---

### 5. Some classes are structurally more difficult

Even though every category contains the same number of samples, classification difficulty varies substantially across classes.

For DeBERTa at L4:

* **Healthy Living:** 0.92 F1
* **Queer Voices:** 0.90 F1
* **Parenting:** 0.90 F1
* **Entertainment:** 0.69 F1
* **Travel:** 0.65 F1

Entertainment and Travel remain among the most difficult categories across multiple context levels.

This suggests that the difficulty is not caused by class imbalance, but by broader and more overlapping category semantics.

---

## Repository Structure

```text
short-text-news-classification/
│
├── README.md
├── Report.pdf
│
├── stage1_logistic/
│   │
│   ├── Logistics+bert_embedding/
│   │   ├── Logistics+bert_embedding.ipynb
│   │   ├── logistic_bert_fold_results.csv
│   │   ├── logistic_bert_summary_results.csv
│   │   └── logistic_bert_per_class_f1_results.csv
│   │
│   └── TF-IDF/
│       └── TF-idf+pos/
│           ├── logistic_TF_IDF_TF_IDF+POS.ipynb
│           ├── logreg_fold_results.csv
│           ├── logreg_summary_results.csv
│           └── logreg_per_class_f1_results.csv
│
├── stage2_BiLSTM/
│   │
│   ├── BiLSTM + Word2Vec/
│   │   ├── BiLSTM+Word2Vec.ipynb
│   │   ├── bilstm_word2vec_fold_results_all_levels.csv
│   │   ├── bilstm_word2vec_summary_results_all_levels.csv
│   │   └── bilstm_word2vec_per_class_f1_results_all_levels.csv
│   │
│   └── BiLSTM + frozen BERT/
│       ├── BiLSTM+Bert_L1L2.ipynb
│       ├── BiLSTM+Bert_L3L4.ipynb
│       ├── bilstm_precomputed_frozen_bert_fold_results_all_levels.csv
│       ├── bilstm_precomputed_frozen_bert_summary_results_all_levels.csv
│       └── bilstm_precomputed_frozen_bert_per_class_f1_results_all_levels.csv
│
└── stage3_DeBERTa fine-tuning/
    │
    ├── debert_adjust_learning_rate/
    │   ├── v1/
    │   │   ├── L1_debert/
    │   │   ├── L2_debert/
    │   │   ├── L3_debert/
    │   │   └── L4_debert/
    │   │
    │   └── v2/
    │       ├── L3_debert/
    │       └── L4_debert/
    │
    ├── debert_model/
    │   ├── L1_bert_V3/
    │   ├── L2_bert_v3/
    │   ├── L3_bert_v3/
    │   └── L4_bert_v3/
    │
    └── debert_result/
        ├── final_result/
        │   ├── debert_all_levels_summary.csv
        │   └── debert_all_levels_per_class_f1.csv
        │
        └── procedure/
            └── debert_all_levels_fold_progress.csv
```

### Stage 1 — Logistic Regression

Contains the linear baseline experiments using:

* TF-IDF
* TF-IDF + POS
* Frozen BERT sentence representations

### Stage 2 — BiLSTM

Contains the intermediate-complexity neural models using:

* Word2Vec
* Frozen BERT token embeddings

### Stage 3 — DeBERTa

Contains:

* preliminary learning-rate experiments;
* final DeBERTa fine-tuning for L1–L4;
* fold-level results;
* aggregated metrics; and
* per-class evaluation results.

---

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Scikit-learn
* BERT
* DeBERTa
* BiLSTM
* Word2Vec
* TF-IDF
* NLTK
* Pandas
* NumPy

---

## Reproducibility

All data-dependent preprocessing is fitted only on the relevant training partitions to prevent data leakage.

This includes:

* TF-IDF vectorizers
* POS feature standardisation
* Word2Vec vocabulary and weights

The same outer 10-fold splits are used across the three main model families to provide a controlled comparison.

The repository also includes fold-level and aggregated result files, allowing the reported results to be inspected without rerunning all computationally expensive nested cross-validation experiments.

## Installation

Install the dependencies required for the final architecture:

```bash
pip install -r requirements.txt
```

---

## Full Report

For the complete methodology, experimental design, statistical evaluation, results, discussion, limitations, and references:

📄 **[View the full report](https://github.com/Mark860787/short-text-news-classification/blob/main/Report.pdf)**

