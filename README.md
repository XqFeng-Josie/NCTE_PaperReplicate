# NCTE Paper Replication

Replication of: [The NCTE transcripts: A dataset of elementary math classroom transcripts](https://arxiv.org/pdf/2211.11772.pdf)  
*Demszky, D., & Hill, H. (2022). arXiv preprint arXiv:2211.11772*

## 📊 Dataset Description

**Data Access**: Fill out the form to get dataset access: https://forms.gle/1yWybvsjciqL8Y9p8

| File | Description |
|------|-------------|
| `ncte_single_utterances.csv` | A csv file containing all utterances from the transcript dataset. The `OBSID` column represents the unique ID for the transcript, and the `NCTETID` represents the teacher ID, which are mappable to metadata. `comb_idx` represents a unique ID for each utterance (concatenation of and `turn_idx`), which is mappable to turn-level annotations. |
| `student_reasoning.csv` | Turn-level annotations for `student_reasoning`. The annotations are binary.|
| `paired_annotations.csv` | Turn-level annotations for `student_on_task`, `teacher_on_task`, `high_uptake`, focu`sing_question, using majority rater labels. The annotation protocol is included under the coding schemes folder. |

**Metadata**:  The transcripts are associated with metadata, including observation scores, value added measures and student questionnaire responses. The metadata and additional documentation are available on [ICPSR](https://www.icpsr.umich.edu/web/ICPSR/studies/36095). You can use the `OBSID` variable and the `NCTETID` variables to map transcript data to the metadata.


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

#### calculation code is [regression.ipynb](regression.ipynb)

#### Variables Explanation
- **Teacher VA**: Teacher Value-Added measures
- **MQI**: Mathematical Quality of Instruction scores
- **CLASS Dimensions**: 
  - CLINSTD (Instructional Dialogue)
  - CLTS (Teacher Sensitivity) 
  - CLRSP (Regard for Student Perspectives)
  - CLPC (Positive Climate)

#### Results Comparison
|             x              |  teacher_va_coef  |   teacher_va_std |  MQI_coef  |   MQI_std |  clinstd_coef  |   clinstd_std |  clts_coef  |   clts_std |  clrsp_coef  |   clrsp_std |  clpc_coef  |   clpc_std |
|:--------------------------:|:-----------------:|-----------------:|:----------:|----------:|:--------------:|--------------:|:-----------:|-----------:|:------------:|------------:|:-----------:|-----------:|
| Focusing Question(0_Paper) |      0.121*       |           -0.05  |  0.117**   |    -0.032 |    0.083**     |        -0.026 |   0.089**   |     -0.019 |   0.058**    |      -0.017 |   0.079**   |     -0.017 |
|  Focusing Question(1_Our)  |      0.2633       |            0.121 |   0.1316   |     0.15  |     0.1632     |         0.13  |   -0.0926   |      0.138 |    0.1151    |       0.132 |   0.0484    |      0.134 |
| Student Reasoning(0_Paper) |      0.191*       |           -0.091 |  0.313**   |    -0.066 |    0.246**     |        -0.05  |   0.144**   |     -0.031 |   0.173**    |      -0.035 |   0.120**   |     -0.035 |
|  Student Reasoning(1_Our)  |      -0.0671      |            0.127 |   0.4362   |     0.133 |     0.3757     |         0.105 |   -0.0165   |      0.105 |    0.1629    |       0.123 |   -0.1266   |      0.116 |
|   Student Turn(0_Paper)    |       1.044       |           -1.357 |   -0.047   |    -0.528 |     0.718      |        -0.669 |    0.214    |     -0.574 |    0.125     |      -0.485 |   -0.172    |     -0.56  |
|    Student Turn(1_Our)     |      -0.0798      |            0.416 |   1.0238   |     0.365 |    -0.0079     |         0.386 |   -0.6615   |      0.376 |   -0.0051    |       0.359 |   -1.2814   |      0.364 |
|   Student Word(0_Paper)    |       0.359       |           -0.792 |   0.721+   |    -0.413 |     1.132*     |        -0.541 |    0.001    |     -0.325 |    0.469     |      -0.395 |    0.322    |     -0.387 |
|    Student Word(1_Our)     |      0.4780       |            0.516 |   0.7003   |     0.449 |     1.1606     |         0.509 |   -0.0752   |      0.337 |    0.5011    |       0.387 |   0.0230    |      0.42  |
|  Student on Task(0_Paper)  |      0.038+       |           -0.02  |   0.022*   |    -0.01  |    0.032**     |        -0.011 |   0.033**   |     -0.008 |   0.024**    |      -0.007 |   0.036**   |     -0.007 |
|   Student on Task(1_Our)   |      0.3425       |            0.12  |   0.3318   |     0.126 |     0.2560     |         0.129 |   0.2543    |      0.12  |   -0.1072    |       0.126 |   0.2176    |      0.115 |
|  Teacher Uptake(0_Paper)   |      0.234*       |           -0.104 |  0.233**   |    -0.086 |    0.198**     |        -0.072 |   0.132**   |     -0.035 |   0.164**    |      -0.044 |   0.115**   |     -0.036 |
|   Teacher Uptake(1_Our)    |      0.0200       |            0.087 |   0.2100   |     0.087 |     0.1257     |         0.1   |   0.1312    |      0.093 |   -0.0122    |       0.093 |   0.1017    |      0.09  |
|  Teacher on Task(0_Paper)  |      0.038+       |           -0.02  |   0.021*   |    -0.01  |    0.030**     |        -0.01  |   0.034**   |     -0.008 |   0.024**    |      -0.007 |   0.035**   |     -0.007 |
|   Teacher on Task(1_Our)   |      0.3264       |            0.136 |   0.3604   |     0.132 |     0.2410     |         0.144 |   0.2245    |      0.119 |   -0.1141    |       0.133 |   0.1910    |      0.128 |
|  z_Observations(0_Paper)   |        523        |          nan     |    1557    |   nan     |      1554      |       nan     |    1554     |    nan     |     1554     |     nan     |    1554     |    nan     |
*Note: Paper results show coefficient significance levels (+p<0.1, *p<0.05, **p<0.01). Our results show raw coefficients.*

## TODO
 1. Correct the results of the second experiment...

## Reference
https://github.com/ddemszky/classroom-transcript-analysis