import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings

def cross_validation(x_train, y_train, params, n_splits=5, random_state=2024):

    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    split_data = kf.split(x_train)

    rmse_results = []
    mape_results = []

    for train_index, val_index in split_data:
        X_train_fold, X_val_fold = x_train.iloc[train_index], x_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

        model = lgb.train(params, train_data, valid_sets=[train_data, val_data])

        y_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)

        rmse = mean_squared_error(y_val_fold, y_pred, squared=False)
        mape = mean_absolute_percentage_error(y_val_fold, y_pred)

        rmse_results.append(rmse)
        mape_results.append(mape)

    print(f'Cross-validation RMSE scores: {rmse_results}')
    print(f'Mean cross-validation RMSE: {sum(rmse_results) / len(rmse_results)}')
    print(f'Cross-validation MAPE scores: {mape_results}')
    print(f'Mean cross-validation MAPE: {sum(mape_results) / len(mape_results)}')
    
    rmse_df = pd.DataFrame(rmse_results, columns=['RMSE'])
    mape_df = pd.DataFrame(mape_results, columns=['MAPE'])

    return rmse_df,mape_df


def train_and_evaluate_model(x_train, y_train, x_test, y_test, best_params, num_boost_round=1000):

    rmse_main_results = []
    mape_main_results = []
    
    # Train the model
    train_data = lgb.Dataset(x_train, label=y_train)
    model = lgb.train(best_params, train_data, num_boost_round=num_boost_round, valid_sets=[train_data])

    # Predict on the test set
    y_pred = model.predict(x_test, num_iteration=model.best_iteration)

    # Calculate RMSE and MAPE
    rmse_main = mean_squared_error(y_test, y_pred, squared=False)
    mape_main = mean_absolute_percentage_error(y_test, y_pred)
    
    rmse_main_results.append(rmse_main)
    mape_main_results.append(mape_main)
    
    rmse_df = pd.DataFrame(rmse_main_results, columns=['RMSE'])
    mape_df = pd.DataFrame(mape_main_results, columns=['MAPE'])
    
    print(f'RMSE on test set: {rmse_main_results}')
    print(f'MAPE on test set: {mape_main}')

    return rmse_df,mape_df,y_pred


def save_dataframe_to_csv(df, base_dir, sub_dir, file_name):
    file_path = os.path.abspath(os.path.join(base_dir, sub_dir, file_name))
    df.to_csv(file_path, index=False)
    print(f'Saved {file_name}')
