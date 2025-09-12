# NCTE Paper Replication

Replication of: [The NCTE transcripts: A dataset of elementary math classroom transcripts](https://arxiv.org/pdf/2211.11772.pdf)  
*Demszky, D., & Hill, H. (2022). arXiv preprint arXiv:2211.11772*

## 📊 Dataset Description

**Data Access**: Fill out the form to get dataset access: https://forms.gle/1yWybvsjciqL8Y9p8

| File | Description |
|------|-------------|
| `ncte_single_utterances.csv` | All utterance data with OBSID (transcript ID), NCTETID (teacher ID), comb_idx (utterance ID) |
| `student_reasoning.csv` | Student reasoning annotations (binary) |
| `paired_annotations.csv` | Turn-level annotations: student_on_task, teacher_on_task, high_uptake, focusing_question |

**Metadata**: Available through [ICPSR](https://www.icpsr.umich.edu/web/ICPSR/studies/36095) for observation scores, value-added measures, etc.

## 🚀 Quick Start

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Training (Exp1)

You can use the `run_classifier.py` script to train turn-level classifiers like the ones described in the paper.

```bash
# Run all experiments
bash run.sh

# Or run specific classifier
python run_classifier.py \
  --train_data=data/paired_annotations.csv \
  --dev_split_size=0.2 \
  --num_train_epochs=5 \
  --text_cols=student_text,teacher_text \
  --label_col=focusing_question \
  --balance_labels \
  --cv \
  --predict_index_col=exchange_idx
```

**Training Options:**
- `--cv`: Enable 5-fold cross-validation
- `--train`: Train without cross-validation
- `--predict`: Run prediction on our data
- `--balance_labels`: Balance training labels

### 3. Merge Results
```bash
# Merge k-fold cross-validation results
python analyze_kfold_best_models.py --kfold_dir outputs/roberta/ --all
```

## 🔬 Experimental Results

### Part 1: Classification Model Performance

Performance comparison between original paper and our replication using 5-fold cross-validation:

**diff = our - paper**

| Measure | Accuracy |  |  | Precision |  |  | Recall |  |  | F1 |  |  |
|---------|----------|---|---|-----------|---|---|--------|---|---|----|----|---|
|         | paper | our | diff | paper | our | diff | paper | our | diff | paper | our | diff |
| Student on Task | 0.902 | 0.908 | 0.56% | 0.952 | 0.954 | 0.16% | 0.931 | 0.937 | 0.61% | 0.942 | 0.945 | 0.32% |
| Teacher on Task | 0.867 | 0.875 | 0.82% | 0.932 | 0.942 | 0.96% | 0.914 | 0.915 | 0.07% | 0.923 | 0.928 | 0.49% |
| Teacher Uptake | 0.768 | 0.783 | 1.52% | 0.719 | 0.718 | -0.14% | 0.674 | 0.683 | 0.89% | 0.688 | 0.698 | 1.03% |
| Focusing Question | 0.856 | 0.853 | -0.33% | 0.474 | 0.514 | 4.03% | 0.538 | 0.527 | -1.10% | 0.501 | 0.517 | 1.63% |
| Student Reasoning | 0.863 | 0.872 | 0.89% | 0.644 | 0.700 | 5.55% | 0.666 | 0.701 | 3.51% | 0.651 | 0.695 | 4.45% |

#### Key Findings
- **Overall Performance**: Our replication achieves comparable or better performance across all tasks
- **Best Improvements**: Student Reasoning shows the largest improvements (+5.6% precision, +4.4% F1)
- **Consistent Results**: Most metrics show small positive improvements, indicating successful replication

### Part 2: Regression Analysis Results

Correlation analysis between classroom discourse measures and teaching quality indicators, comparing our replication with the original paper (Table 5).

#### Variables Explanation
- **Teacher VA**: Teacher Value-Added measures
- **MQI**: Mathematical Quality of Instruction scores
- **CLASS Dimensions**: 
  - CLINSTD (Instructional Dialogue)
  - CLTS (Teacher Sensitivity) 
  - CLRSP (Regard for Student Perspectives)
  - CLPC (Positive Climate)

#### Results Comparison
|    | x                        | teacher_va_coef   |   teacher_va_std | MQI_coef   |   MQI_std | clinstd_coef   |   clinstd_std | clts_coef   |   clts_std | clrsp_coef   |   clrsp_std | clpc_coef   |   clpc_std |
|---:|:-------------------------|:------------------|-----------------:|:-----------|----------:|:---------------|--------------:|:------------|-----------:|:-------------|------------:|:------------|-----------:|
|  0 | Focusing Question(Our)   | 0.23              |            0.123 | 0.166      |     0.126 | 0.165          |         0.121 | -0.092      |      0.121 | 0.131        |       0.126 | 0.084       |      0.121 |
|  1 | Focusing Question(Paper) | 0.121*            |           -0.05  | 0.117**    |    -0.032 | 0.083**        |        -0.026 | 0.089**     |     -0.019 | 0.058**      |      -0.017 | 0.079**     |     -0.017 |
|  2 | Student Reasoning(Our)   | -0.07             |            0.125 | 0.443      |     0.134 | 0.373          |         0.118 | -0.016      |      0.119 | 0.16         |       0.123 | -0.12       |      0.117 |
|  3 | Student Reasoning(Paper) | 0.191*            |           -0.091 | 0.313**    |    -0.066 | 0.246**        |        -0.05  | 0.144**     |     -0.031 | 0.173**      |      -0.035 | 0.120**     |     -0.035 |
|  4 | Student Turn(Our)        | -0.003            |            0.327 | 1.004      |     0.329 | -0.005         |         0.332 | -0.686      |      0.331 | -0.016       |       0.332 | -1.314      |      0.33  |
|  5 | Student Turn(Paper)      | 1.044             |           -1.357 | -0.047     |    -0.528 | 0.718          |        -0.669 | 0.214       |     -0.574 | 0.125        |      -0.485 | -0.172      |     -0.56  |
|  6 | Student Word(Our)        | 0.543             |            0.367 | 0.642      |     0.365 | 1.193          |         0.368 | -0.113      |      0.367 | 0.556        |       0.368 | -0.021      |      0.367 |
|  7 | Student Word(Paper)      | 0.359             |           -0.792 | 0.721+     |    -0.413 | 1.132*         |        -0.541 | 0.001       |     -0.325 | 0.469        |      -0.395 | 0.322       |     -0.387 |
|  8 | Student on Task(Our)     | 0.325             |            0.124 | 0.316      |     0.125 | 0.261          |         0.12  | 0.27        |      0.119 | -0.082       |       0.126 | 0.243       |      0.12  |
|  9 | Student on Task(Paper)   | 0.038+            |           -0.02  | 0.022*     |    -0.01  | 0.032**        |        -0.011 | 0.033**     |     -0.008 | 0.024**      |      -0.007 | 0.036**     |     -0.007 |
| 10 | Teacher Uptake(Our)      | -0.009            |            0.086 | 0.235      |     0.088 | 0.119          |         0.085 | 0.123       |      0.084 | -0.01        |       0.088 | 0.112       |      0.085 |
| 11 | Teacher Uptake(Paper)    | 0.234*            |           -0.104 | 0.233**    |    -0.086 | 0.198**        |        -0.072 | 0.132**     |     -0.035 | 0.164**      |      -0.044 | 0.115**     |     -0.036 |
| 12 | Teacher on Task(Our)     | 0.312             |            0.133 | 0.345      |     0.133 | 0.244          |         0.128 | 0.241       |      0.127 | -0.088       |       0.134 | 0.219       |      0.128 |
| 13 | Teacher on Task(Paper)   | 0.038+            |           -0.02  | 0.021*     |    -0.01  | 0.030**        |        -0.01  | 0.034**     |     -0.008 | 0.024**      |      -0.007 | 0.035**     |     -0.007 |
| 14 | z_Observations(Paper)    | 523               |          nan     | 1557       |   nan     | 1554           |       nan     | 1554        |    nan     | 1554         |     nan     | 1554        |    nan     |

*Note: Paper results show coefficient significance levels (+p<0.1, *p<0.05, **p<0.01). Our results show raw coefficients.*

## TODO
 1. Correct the results of the second experiment...