README

Project: Group 8 - COMP90051 Statistical Machine Learning Group Project
Topic: Short-text news classification under reduced textual context

This archive contains the code and result files for the project. The main research question is how textual feature construction under reduced context affects classifier generalization in short-text news topic classification. The project compares models with different levels of complexity: a logistic regression baseline, a BiLSTM model, and a DeBERTa fine-tuning model.

Dataset

The project uses the News Category Dataset by Rishabh Misra. It can be downloaded from Kaggle:

https://www.kaggle.com/datasets/rmisra/news-category-dataset

Original dataset file:

News_Category_Dataset_v3.json

The dataset itself is not included in this submission archive. Please download the JSON file from Kaggle and place it in the expected local data path before running the notebooks. If the dataset is stored in a different location, update the data-loading path in the corresponding notebook.

For this project, the experiments use a balanced short-text classification subset constructed from the original dataset. The input text is converted into four context levels:

L1: truncated headline only
L2: full headline only
L3: headline plus partial short description
L4: headline plus full short description

Folder structure

1. stage1_logistic/
   Contains the Stage 1 logistic regression experiments and result files. This is the simpler model class used in the report.

   The logistic regression experiments include two input feature variants:

   1.1 Logistics+bert_embedding/
       Uses frozen BERT embeddings as fixed text features for logistic regression.
       BERT is used only as a feature extractor in this stage; the logistic regression classifier is trained on the extracted embeddings.

       Main notebook:
       * Logistics+bert_embedding.ipynb

       Result files:
       * logistic_bert_fold_results.csv
       * logistic_bert_summary_results.csv
       * logistic_bert_per_class_f1_results.csv

   1.2 TF-IDF/TF-idf+pos/
       Contains the sparse-feature logistic regression experiments.
       This folder includes the TF-IDF baseline and the TF-IDF plus POS-feature variant.

       Main notebook:
       * logistic_TF_IDF_TF_IDF+POS.ipynb

       Result files:
       * logreg_fold_results.csv
       * logreg_summary_results.csv
       * logreg_per_class_f1_results.csv

   Notes:
   - The notebooks assume that News_Category_Dataset_v3.json has already been downloaded locally.
   - If the dataset path differs from the development environment, update the data-loading path in the notebook before running.
   - The result CSV files are included so that the reported logistic regression metrics can be inspected without rerunning the full nested cross-validation.

2. stage2_BiLSTM/
   Contains the Stage 2 BiLSTM-based experiments and result files. This is the intermediate-complexity model class used in the report.

   The BiLSTM experiments include two input representation variants:

   2.1 BiLSTM + Word2Vec
       Uses 100-dimensional skip-gram Word2Vec embeddings trained only on the training text within each fold.
       The tuned hyperparameter is the BiLSTM hidden dimension, using the grid [64, 128, 256].
       The model is trained with Adam and dropout 0.3.

       Main notebook:
       * BiLSTM+Word2Vec.ipynb

       Result files:
       * bilstm_word2vec_fold_results_all_levels.csv
       * bilstm_word2vec_summary_results_all_levels.csv
       * bilstm_word2vec_per_class_f1_results_all_levels.csv

   2.2 BiLSTM + frozen BERT
       Uses frozen bert-base-uncased token-level embeddings as sequential BiLSTM inputs.
       BERT is used only as a frozen feature extractor; no BERT parameters are fine-tuned in this stage.
       The tuned hyperparameter is the BiLSTM hidden dimension, using the grid [128, 256, 384].
       The model is trained with Adam and dropout 0.3.

       Main notebooks:
       * BiLSTM+Bert_L1L2.ipynb
       * BiLSTM+Bert_L3L4.ipynb

       Result files:
       * bilstm_precomputed_frozen_bert_fold_results_all_levels.csv
       * bilstm_precomputed_frozen_bert_summary_results_all_levels.csv
       * bilstm_precomputed_frozen_bert_per_class_f1_results_all_levels.csv

   Notes:
   - The notebooks assume that News_Category_Dataset_v3.json has already been downloaded locally.
   - If the dataset path differs from the development environment, update the data-loading path in the notebook before running.
   - The result CSV files are included so that the reported BiLSTM metrics can be inspected without rerunning the full nested cross-validation.

3. stage3_DeBERTa fine-tuning/
   Contains the Stage 3 DeBERTa fine-tuning experiments and result files. This is the complex contextual representation model used in the report.

   3.1 debert_adjust_learning_rate/
       Contains preliminary DeBERTa learning-rate adjustment experiments. These folders were used to test whether the final learning-rate range should be shifted upward.

       - v1/
         First learning-rate tuning version.
         Learning-rate grid used in this version: [5e-6, 1e-5, 2e-5]

         * L1_debert/
           Context Level 1: truncated headline only.
           Learning-rate grid: [5e-6, 1e-5, 2e-5]

         * L2_debert/
           Context Level 2: full headline only.
           Learning-rate grid: [5e-6, 1e-5, 2e-5]

         * L3_debert/
           Context Level 3: headline plus partial short description.
           Learning-rate grid: [5e-6, 1e-5, 2e-5]

         * L4_debert/
           Context Level 4: headline plus full short description.
           Learning-rate grid: [5e-6, 1e-5, 2e-5]

       - v2/
         Second learning-rate tuning version.
         Learning-rate grid used in this version: [1e-5, 2e-5, 5e-5]

         * L3_debert/
           Context Level 3: headline plus partial short description.
           Learning-rate grid: [1e-5, 2e-5, 5e-5]

         * L4_debert/
           Context Level 4: headline plus full short description.
           Learning-rate grid: [1e-5, 2e-5, 5e-5]

   3.2 debert_model/
       Contains the final DeBERTa model experiments used for the report. Each subfolder corresponds to one input context level. The final learning-rate grid was shifted to [2e-5, 5e-5, 8e-5] after the preliminary learning-rate adjustment trials.

       - L1_bert_V3/
         Context Level 1: truncated headline only.
         Final learning-rate grid: [2e-5, 5e-5, 8e-5]
         Most frequent selected learning rate: 5e-5
         Selection count: 5e-5 selected in 10/10 outer folds

       - L2_bert_v3/
         Context Level 2: full headline only.
         Final learning-rate grid: [2e-5, 5e-5, 8e-5]
         Most frequent selected learning rate: 5e-5
         Selection count: 5e-5 selected in 8/10 outer folds; 2e-5 selected in 2/10 outer folds

       - L3_bert_v3/
         Context Level 3: headline plus partial short description.
         Final learning-rate grid: [2e-5, 5e-5, 8e-5]
         Most frequent selected learning rate: 5e-5
         Selection count: 5e-5 selected in 6/10 outer folds; 2e-5 selected in 4/10 outer folds

       - L4_bert_v3/
         Context Level 4: headline plus full short description.
         Final learning-rate grid: [2e-5, 5e-5, 8e-5]
         Most frequent selected learning rate: 5e-5
         Selection count: 5e-5 selected in 9/10 outer folds; 2e-5 selected in 1/10 outer folds

   3.3 debert_result/
       Contains collected DeBERTa result files used for analysis and report writing.

       - final_result/
         Stores final aggregated result CSV files, such as all-level summary metrics and per-class F1 scores.
         Example files:
         * debert_all_levels_summary.csv
         * debert_all_levels_per_class_f1.csv

       - procedure/
         Stores intermediate procedure files, such as fold-level progress records from cross-validation.
         Example file:
         * debert_all_levels_fold_progress.csv

Notes

- The dataset is not included in this archive.
- The notebooks assume that the raw Kaggle JSON dataset has already been downloaded locally.
- If the local path differs from the path used during development, modify the dataset path in the notebook before running it.
- The DeBERTa notebooks contain the Stage 3 implementation and output files for the four context levels.
