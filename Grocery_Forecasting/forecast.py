# ====================================================================
# Import Libraries
# ====================================================================

import os
import sys

# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(src_path)

import pandas as pd
import numpy as np 
from datetime import datetime

from config import *
from features import *
from modelling import *
from visualize import *

print ("***************************************************************************************")
print("E-Grocery Order Forecasting - Started : ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print ("****************************************************************************************")


# ====================================================================
# File Read
# ====================================================================

# Construct the relative path
base_dir = os.path.dirname(__file__)

# Construct the absolute path to the file
file_path = os.path.abspath(os.path.join(base_dir, 'data', 'raw', 'train.csv'))
df_train = pd.read_csv(file_path)

file_path = os.path.abspath(os.path.join(base_dir, 'data', 'raw', 'test.csv'))
df_test = pd.read_csv(file_path)

file_path = os.path.abspath(os.path.join(base_dir,  'data', 'raw', 'train_calendar.csv'))
calendar_train = pd.read_csv(file_path)

file_path = os.path.abspath(os.path.join(base_dir, 'data', 'raw', 'test_calendar.csv'))
calendar_test = pd.read_csv(file_path)

df_train['date'] = df_train['date'].astype('datetime64[ns]')
calendar_train['date'] = calendar_train['date'].astype('datetime64[ns]')
df_test['date'] = df_test['date'].astype('datetime64[ns]')
calendar_test['date'] = calendar_test['date'].astype('datetime64[ns]')

# Drop Columns - id, warehouse_limited
df_train = drop_columns(df_train, DROP_COLS)
df_test = drop_columns(df_test, DROP_COLS)

calendar_train = drop_columns(calendar_train, DROP_COLS)
calendar_test= drop_columns(calendar_test, DROP_COLS)

print("File Reading - Completed : ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"\n")

# ====================================================================
# Feature Engineering
# ====================================================================

# Identify Warehouse working & non-working days
calendar_train = fill_warehouse_closed_days(calendar_train, df_train)
calendar_test = fill_warehouse_closed_days(calendar_test, df_test)


# Merge train & test. 
# Segregate features that have to be engineered as "features" & 
# "labels" are columns that will be later target encoded and it has target variable 'Orders' as well.
features, labels = merge_information_and_train_test_set(calendar_train, df_train, calendar_test, df_test)


# Calculates consecutive closed days ahead.
# If Warehouse is closed between 25 Dec to 1 Jan. The feature creates assigns value- 7 for 25th Dec, 6 for 26th Dec 
# until 1 for 1st Jan. This gives warehouse and business heads up of remainng holidays.

features = calculate_continuous_closed_days(features)
features.loc[features['continuous_holiday'] > 10, ['continuous_holiday']] = 0

# Calculates how many days the warehouse was shutdown before today.
# Again this should give inventory stocking, resource mgmt headsup to the organisation.

features = calculate_continuous_shut_down_before(features)
features = features.drop('warehouse_work_day', axis=1)

# Holiday features

features = away_from_holiday(features,4)
features = after_holiday(features, 1)

# Date features 
# Basic - year, month, weekday, season
# Advanced - month-start/end, quarter-start/end,day_num,quarter,week_of_year etc

features = split_date(features,'date')
features = get_advanced_date_information(features,'date')

# Calculate - orders_ratio_of_week (from orders and orders_sum_week) => both at week and month levels
labels = make_orders_ratio(features, labels)


# Data period :  15-Feb-22 to 15-Feb-24
start = pd.to_datetime('2022-02-15')
end = pd.to_datetime('2024-03-15')

features = features.loc[(features['date'] >= start) & (features['date'] <= end)].reset_index(drop=True)
labels = pd.merge(features[PRIMARY_KEY], labels, how='left', on=PRIMARY_KEY).reset_index(drop=True)


# Encoding -  Binary, One-hot and Target based on columns

features = target_encoding(features, labels, HOLIDAY_NAME_COLS, n_splits=5)
features = target_encoding(features, labels, CATE_COLS, n_splits=5)

features = one_hot_pipeline(features, DATE_COLS)
features = one_hot_pipeline(features, WEEKDAY_COLS)
features = binary_encode(features, BOOL_COLS)
features = linear_encoding(features, LINEAR_COLS)

# Drop redundant features

all_features = list(set(HOLIDAY_NAME_COLS) | set(DATE_COLS) | set(WEEKDAY_COLS) | set(BOOL_COLS) | set(CATE_COLS) | set(LINEAR_COLS))
all_features = list(set(all_features) - set(PRIMARY_KEY))
# print("Dropping these features : ",all_features)
features = features.drop(all_features, axis=1)


print("Feature Engineering - Completed : ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"\n")
print ("***************************************************************************************")


# ====================================================================
# Modelling - Data preparation (Train, Test)
# ====================================================================

# Merge Features & Label dataframes. Drop records when there are no orders.

features = pd.merge(features, labels[['date','warehouse','orders']], how='left',on=['date','warehouse']).reset_index(drop=True)
features = features[features['orders'].notna()].reset_index(drop=True)

train = features.loc[features['date'] <= '2024-02-14'].reset_index(drop=True)
test = features.loc[(features['date'] >= '2024-02-15')].reset_index(drop=True)

x_train = train.copy()
x_test = test.copy()

x_train = x_train.drop(['date', 'warehouse', 'train','orders'], axis=1)
x_test = x_test.drop(['date', 'warehouse', 'train','orders'], axis=1)

y_train = train['orders']
y_test = test['orders']

print(f'train: {x_train.shape}, test: {x_test.shape}')

# ====================================================================
# Modelling - Cross Validation
# ====================================================================

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

rmse_cv_results, mape_cv_results = cross_validation(x_train, y_train, params)


print("Model Crossvalidation - Completed : ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"\n")
print ("***************************************************************************************")


# ====================================================================
# Modelling - Train and prediction
# ====================================================================

print("Train period : ", min(train.date), max(train.date))
print("Test period : ", min(test.date), max(test.date),"\n")

best_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

rmse_main, mape_main,test_preds = train_and_evaluate_model(x_train, y_train, x_test, y_test, best_params)

test['preds'] = test_preds
df_test_preds= test[['date','warehouse','orders','preds']].sort_values(by=['date','warehouse']).reset_index(drop=True)


print("Model Training & Inference is completed : ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"\n")
print ("***************************************************************************************")


# ====================================================================
# Visualization
# ====================================================================

visualization(df_test_preds)


print("Test results - Actuals vs Forecast - Visualized : ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print ("***************************************************************************************")

# ====================================================================
# File Backup
# ====================================================================

# Example usage:
save_dataframe_to_csv(train, base_dir, 'data/processed', 'train_data.csv')
save_dataframe_to_csv(test, base_dir, 'data/processed', 'test_data.csv')
save_dataframe_to_csv(rmse_cv_results, base_dir, 'data/metrics', 'rmse_cv_results.csv')
save_dataframe_to_csv(mape_cv_results, base_dir, 'data/metrics', 'mape_cv_results.csv')
save_dataframe_to_csv(rmse_main, base_dir, 'data/metrics', 'rmse_main.csv')
save_dataframe_to_csv(mape_main, base_dir, 'data/metrics', 'mape_main.csv')
save_dataframe_to_csv(df_test_preds, base_dir, 'data/output', 'df_test_preds.csv')


print("\n")
print ("***************************************************************************************")
print("E-Grocery Order Forecasting - Completed : ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print ("****************************************************************************************")




