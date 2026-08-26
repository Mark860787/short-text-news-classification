# Short-Text News Classification under Reduced Context

This project investigates how **textual feature construction under reduced context affects classifier generalization in short-text news classification**.

We compare models with different levels of complexity — **Logistic Regression, BiLSTM, and fine-tuned DeBERTa** — across progressively richer textual context levels.

📄 **[Read the full project report](./report.pdf)**

## Project Overview

Short-text classification is challenging because limited context can remove important semantic information. This project examines whether more sophisticated representations and model architectures can compensate for missing textual context.

The experiments use the **News Category Dataset** and focus on the top 10 categories, with 2,000 samples per class for a balanced dataset of **20,000 news articles**.

Four context levels are constructed:

* **L1:** First 5 words of the headline
* **L2:** Full headline
* **L3:** Headline + first 15 words of the short description
* **L4:** Headline + full short description

## Models and Representations

### 1. Logistic Regression

Three feature representations are evaluated:

* TF-IDF
* TF-IDF + POS features
* Frozen BERT sentence embeddings

### 2. BiLSTM

Two embedding strategies are compared:

* Word2Vec embeddings
* Frozen BERT token embeddings

### 3. DeBERTa

`microsoft/deberta-base` is fine-tuned end-to-end for multi-class news classification.

## Evaluation

All models are evaluated using the same **nested stratified cross-validation framework**:

* 10-fold outer cross-validation
* 3-fold inner cross-validation for hyperparameter selection
* Identical outer folds across model families

Evaluation metrics include:

* Accuracy
* Macro F1
* Weighted F1
* Per-class F1

## Key Results

| Model                             | L1 Macro F1 | L2 Macro F1 | L3 Macro F1 | L4 Macro F1 |
| --------------------------------- | ----------: | ----------: | ----------: | ----------: |
| Logistic Regression + TF-IDF      |       0.562 |       0.690 |       0.716 |       0.735 |
| Logistic Regression + Frozen BERT |       0.570 |       0.702 |       0.732 |       0.750 |
| BiLSTM + Word2Vec                 |       0.188 |       0.325 |       0.435 |       0.489 |
| BiLSTM + Frozen BERT              |       0.558 |       0.688 |       0.721 |       0.733 |
| **Fine-tuned DeBERTa**            |   **0.640** |   **0.767** |   **0.812** |   **0.835** |

### Main Findings

* **More textual context consistently improves classification performance.**
* **Fine-tuned DeBERTa achieves the strongest performance at every context level.**
* Increasing context has a substantially larger effect than changing feature representation within the same classifier.
* **Representation quality can matter more than model complexity:** BiLSTM with Word2Vec performs substantially worse than Logistic Regression with TF-IDF, while replacing Word2Vec with frozen BERT embeddings recovers most of the performance gap.
* Performance gains show **diminishing returns** as context increases. The improvement from L3 to L4 is considerably smaller than earlier context expansions.
* Some categories remain structurally more difficult. In particular, **Entertainment** and **Travel** consistently obtain lower per-class F1 scores despite balanced class sizes.

## Repository Structure

```text
.
├── README.md
├── report.pdf
│
├── stage1_logistic/
│   ├── Logistics+bert_embedding/
│   └── TF-IDF/
│
├── stage2_BiLSTM/
│   ├── BiLSTM + Word2Vec
│   └── BiLSTM + frozen BERT
│
└── stage3_DeBERTa_fine-tuning/
    ├── debert_adjust_learning_rate/
    ├── debert_model/
    └── debert_result/
```

## Dataset

The project uses the **News Category Dataset** by Rishabh Misra.

Dataset:

https://www.kaggle.com/datasets/rmisra/news-category-dataset

The dataset is **not included in this repository**.

After downloading:

```text
News_Category_Dataset_v3.json
```

update the dataset path in the relevant notebooks if necessary.

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

## Reproducibility

All data-dependent preprocessing is fitted only on the corresponding training partitions to prevent data leakage.

The repository includes fold-level and aggregated result files, allowing the reported results to be inspected without rerunning all computationally expensive nested cross-validation experiments.

## Report

For the complete methodology, experimental design, results, discussion, and limitations:

**[View the full report](./report.pdf)**
