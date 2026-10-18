# Titanic model metrics

Training records: 712. Test records: 179. Positive class: survived (1).
These are test results from models fitted on the training records.

| Model / implementation | Accuracy | Precision | Recall | F1 | ROC-AUC | Log-loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive Bayes (scikit-learn) | 0.787709 | 0.720588 | 0.720588 | 0.720588 | 0.840885 | 0.654587 |
| Naive Bayes (NumPy) | 0.787709 | 0.720588 | 0.720588 | 0.720588 | 0.840885 | 0.654587 |
| Logistic Regression (scikit-learn) | 0.804469 | 0.753846 | 0.720588 | 0.736842 | 0.864997 | 0.434158 |
| Logistic Regression (NumPy) | 0.804469 | 0.753846 | 0.720588 | 0.736842 | 0.864865 | 0.434034 |

Majority-class baseline accuracy: 0.6201.

## Confusion matrices

![Naive Bayes: library and NumPy](graphs/naive_bayes_confusion_matrix.png)

![Logistic Regression: library and NumPy](graphs/logistic_regression_confusion_matrix.png)

Rows are true labels; columns are predictions, ordered [0, 1].

## Implementations

The scikit-learn estimators remain as baselines. NumPy model training and prediction are implemented directly in each notebook.
Naive Bayes estimates priors, class means/variances, Gaussian log-likelihoods and normalized probabilities.
NumPy Logistic Regression minimizes mean binary cross-entropy plus ||w||^2/(2*C*n), with an unpenalized intercept.
It uses full-batch gradient descent: C=1, learning rate 0.1, maximum 50,000 steps, gradient tolerance 1e-7.
The library LR baseline uses C=1 with its default L-BFGS solver and max_iter=1000.
Scikit-learn is still used for shared preparation, evaluation metrics and confusion-matrix display.

![NumPy Logistic Regression training loss](graphs/logistic_regression_training_loss.png)

## Findings

Naive Bayes accuracy: scikit-learn 78.77%, NumPy 78.77%.
Log-loss: scikit-learn 0.654587, NumPy 0.654587.

Logistic Regression accuracy: scikit-learn 80.45%, NumPy 80.45%.
Log-loss: scikit-learn 0.434158, NumPy 0.434034.

All implementations use the same fixed records and training-only preparation. Optimizer convergence is checked on training data; test scores are not used to choose settings.
The test partition was inspected during earlier development; this is a fixed comparison, not a new independent test.
A single historical sample does not establish that one model is universally better.
