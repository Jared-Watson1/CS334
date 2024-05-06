# Examining Bias in Loan Approval Processes with Machine Learning: Lending Club Dataset

Jared Watson, Shiv Desai

## Setup

1. Download the datasets [here](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv)
2. Run the `preprocess_data.py` script on the downloaded `loan.csv` file from previous step
3. Run the `feature_selection.py` script on the outputted file from previous step
   1. Get statistics regarding feature correlation and importance. Statistics will be saved to a json
4. Run the `split_data.py` script which will use the newly created json to select best features for prediction
   1. then data will be split into training and testing sets (75/25)

## Results / Notes

- Used MinMaxScaler for numerical values

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

    accuracy                           0.98    282243
   macro avg       0.78      0.78      0.78    282243
weighted avg       0.98      0.98      0.98    282243
```

# Decision Tree w/ Demographic Features Removed

```bash
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

# KNN Classifer w/ Demographic Features Removed

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
/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
Accuracy: 0.98
Training Logistic Regression (C=100, penalty='l2')...
/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
Accuracy: 0.98
Finished logistic regression training and evaluation. Results saved to logistic_regression.json.
```

- Logistic regression parameter tuning results stored in `logistic_regression.json`

# Logisitic Regression w/ Demographic Features Removed

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
Finished logistic regression training and evaluation. Results saved to logistic_regression.json.
```
