# Titanic model metrics

Training records: 712. Test records: 179. Positive class: survived (1).
These are test results from models fitted on the training records.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Log-loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive Bayes | 0.7877 | 0.7206 | 0.7206 | 0.7206 | 0.8409 | 0.6546 |
| Logistic Regression | 0.8045 | 0.7538 | 0.7206 | 0.7368 | 0.8650 | 0.4342 |

Majority-class baseline accuracy: 0.6201.

## Confusion matrices

![Naive Bayes confusion matrix](graphs/naive_bayes_confusion_matrix.png)

![Logistic Regression confusion matrix](graphs/logistic_regression_confusion_matrix.png)

Rows are true labels; columns are predictions, ordered [0, 1].

## Findings

Naive Bayes accuracy: 78.77%; Logistic Regression accuracy: 80.45%.
LR-minus-NB accuracy difference: 1.68 percentage points.
Survival recall: NB 0.7206, LR 0.7206.
Log-loss (lower is better): NB 0.6546, LR 0.4342.

Both models use the same fixed records and training-only preparation. No parameter search or threshold tuning is used.
The test partition was inspected during earlier development; this is a fixed comparison, not a new independent test.
A single historical sample does not establish that one model is universally better.
