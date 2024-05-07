# Examining Bias in Loan Approval Processes with Machine Learning: Lending Club Dataset

Jared Watson, Shiv Desai

## Setup

1. Download the datasets [here](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv)
2. Run the `split_and_preprocess.py` script on the downloaded `loan.csv` file from previous step
   a. The data will be split into training and testing sets (80/20)
3. Run the `feature_selection.py` script on the outputted file from previous step
   1. Get statistics regarding feature correlation and importance. Statistics will be saved to a JSON
4. Run the `split_data.py` script which will use the newly created json to select best features for prediction
   1. then data will be split into training and testing sets (75/25)
5. Each model script (e.g. `decision_tree.py`, `knn.py`, and `logistic_regression.py`) will run the model and tune parameters
   1. Best parameters will be saved to the relevant JSON file (e.g `model_name_params.json`)
6. `model_evaluation.py` will use each models best parameters and create a report of the performance for each model.

## Results / Notes

- Used MinMaxScaler for numerical values

## Decision Tree w/ Demographic Features

```bash
python3 decision_tree.py
Starting decision tree training...
Preparing feature matrices and target vectors...
Finished preparing data: 6.597320795059204
Starting training...
Finished training: 53.225985050201416
Evaluating model...
Finished evaluating model: 53.568256855010986
Accuracy of the Decision Tree model: 0.98
Classification Report:
               precision    recall  f1-score   support

           0       0.98      0.98      0.98    114955
           1       0.99      0.99      0.99    130325
           3       0.99      0.99      0.99     32537
           4       0.75      0.75      0.75      3287
           5       0.16      0.18      0.17      1139

    accuracy .                          0.98    282243
   macro avg       0.78      0.78      0.78    282243
weighted avg       0.98      0.98      0.98    282243
```

## Decision Tree w/ Demographic Features Removed

```bash
python3 decision_tree.py
Starting decision tree training...
Preparing feature matrices and target vectors...
Finished preparing data: 15.080891132354736
Starting training...
Finished training: 112.60323429107666
Evaluating model...
Finished evaluating model: 113.13399934768677
Accuracy of the Decision Tree model: 0.98
Classification Report:
               precision    recall  f1-score   support

           0       0.98      0.98      0.98     91954
           1       0.99      0.99      0.99    104171
           3       0.97      0.98      0.98     26223
           4       0.75      0.75      0.75      2546
           5       0.16      0.18      0.17       901

    accuracy                           0.98    225795
   macro avg       0.77      0.78      0.77    225795
weighted avg       0.98      0.98      0.98    225795
```

## KNN Classifer w/ Demographic Features

```bash
python3 knn.py
Starting KNN classifier training...
Preparing feature matrices and target vectors...
Training KNN (k=3, metric=euclidean)...
Accuracy: 0.84
Training KNN (k=5, metric=euclidean)...
Accuracy: 0.85
Training KNN (k=7, metric=euclidean)...
Accuracy: 0.85
Training KNN (k=9, metric=euclidean)...
Accuracy: 0.85
Training KNN (k=3, metric=manhattan)...
Accuracy: 0.86
Training KNN (k=5, metric=manhattan)...
Accuracy: 0.87
Training KNN (k=7, metric=manhattan)...
Accuracy: 0.88
Training KNN (k=9, metric=manhattan)...
Accuracy: 0.88
```

## KNN Classifer w/ Demographic Features Removed

```bash
python3 knn.py
Starting KNN classifier training...
Preparing feature matrices and target vectors...
Training KNN (k=3, metric=euclidean)...
Accuracy: 0.85
Training KNN (k=5, metric=euclidean)...
Accuracy: 0.86
Training KNN (k=7, metric=euclidean)...
Accuracy: 0.86
Training KNN (k=9, metric=euclidean)...
Accuracy: 0.86
Training KNN (k=3, metric=manhattan)...
Accuracy: 0.87
Training KNN (k=5, metric=manhattan)...
Accuracy: 0.87
Training KNN (k=7, metric=manhattan)...
Accuracy: 0.89
Training KNN (k=9, metric=manhattan)...
Accuracy: 0.89
```

## Logisitic Regression w/ Demographic Features

```bash
python3 logistic_regression.py
Starting logistic regression training...
Preparing feature matrices and target vectors...
Training Logistic Regression (C=0.01, penalty='l2')...
Accuracy: 0.97
Training Logistic Regression (C=0.1, penalty='l2')...
Accuracy: 0.98
Training Logistic Regression (C=1, penalty='l2')...
Accuracy: 0.98
Training Logistic Regression (C=10, penalty='l2')...
Accuracy: 0.98
Training Logistic Regression (C=100, penalty='l2')...
Accuracy: 0.98
```

```bash
python3 model_evaluation.py
Starting model evaluation...
Loading and preparing data...
Testing model: decision_tree
Model: {model_name}
Best params: {best_params}
/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
Testing model: knn
Model: {model_name}
Best params: {best_params}
Testing model: logistic_regression
Model: {model_name}
Best params: {best_params}
/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
   Accuracy  Precision (Weighted)  Recall (Sensitivity, Weighted)  ...  F1 Score (Weighted)                Model ROC AUC Score
0  0.527007              0.435973                        0.527007  ...             0.450511        decision_tree           1.0
1  0.864532              0.860955                        0.864532  ...             0.851454                  knn           1.0
2  0.978501              0.973393                        0.978501  ...             0.975296  logistic_regression           1.0

[3 rows x 7 columns]
Done. Plotting ROC Curves...
## Logisitic Regression w/ Demographic Features Removed

```bash
Starting logistic regression training...
Preparing feature matrices and target vectors...
Training Logistic Regression (C=0.01, penalty='l2')...
Accuracy: 0.96
Training Logistic Regression (C=0.1, penalty='l2')...
Accuracy: 0.96
Training Logistic Regression (C=1, penalty='l2')...
Accuracy: 0.96
Training Logistic Regression (C=10, penalty='l2')...
Accuracy: 0.96
Training Logistic Regression (C=100, penalty='l2')...
Accuracy: 0.96
```
