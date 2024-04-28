# Examining Bias in Loan Approval Processes with Machine Learning: Lending Club Dataset

Jared Watson, Shiv Desai

## Setup

1. Download the datasets [here](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv)
2. Run the `preprocess_data.py` script on the downloaded `loan.csv` file from previous step
3. Run the `feature_selection.py` script on the outputted file from previous step
   1. Get statistics regarding feature correlation and importance. Statistics will be saved to a json

## Results / Notes

- Used MinMaxScaler for numerical values
