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
