import pandas as pd
from src.config import *
from sklearn.preprocessing import OneHotEncoder

#Drop Columns
def drop_columns(df, cols):
    cols_need_drop = list(set(cols) & set(df.columns))
    if len(cols_need_drop) > 0:
        return df.drop(cols_need_drop, axis=1)
    return df

#Sort data by warehouse-date
def sort_by_warehouse_and_date(df):
    return df.sort_values(['warehouse', 'date']).reset_index(drop=True)


# Identify Warehouse working & non-working days
def fill_warehouse_closed_days(calendar, data):
    if 'warehouse_work_day' in calendar.columns:
        calendar.drop('warehouse_work_day', axis=1, inplace=True)
    df1 = calendar.copy()
    df2 = data[['warehouse', 'date']].copy()
    
    df1 = sort_by_warehouse_and_date(df1)
    df2 = sort_by_warehouse_and_date(df2)
    
    df2['warehouse_work_day'] = True
    df1 = pd.merge(df1, df2, how='left', on=['warehouse', 'date'])
    df1['warehouse_work_day'] = df1['warehouse_work_day'].fillna(False).astype(bool)
    return df1


def merge_information_and_train_test_set(calendar_train, df_train, calendar_test, df_test):
    extra_info = list(set(LABEL1+LABEL2) - set(calendar_train))
    df = pd.merge(calendar_train, df_train[PRIMARY_KEY + extra_info], how='left', on=['warehouse','date'])
    features = df.drop(LABEL1+LABEL2, axis=1)
    labels = df[PRIMARY_KEY+LABEL1+LABEL2]
    features['train'] = True
    calendar_test['train'] = False
    features = pd.concat([features, calendar_test])
    features = sort_by_warehouse_and_date(features)
    labels = sort_by_warehouse_and_date(labels)
    return features, labels


def calculate_continuous_closed_days(df):
    # Create a new column 'warehouse_non_work_day' where 'normal_work' is False (0) or True (1)
    df['warehouse_non_work_day'] = (~df['warehouse_work_day']).astype('int32')
    
    df = df.sort_values(by=['warehouse', 'date']).reset_index(drop=True)
    
    # Calculate continuous holidays (consecutive days ahead) for each warehouse. 
    # Helps Companies know non-working days in advance.
    df['continuous_holiday'] = df.groupby('warehouse')['warehouse_non_work_day'].apply(
        lambda x: x[::-1].groupby((x != x.shift()).cumsum()).cumsum()[::-1] * x
    )
    
    df = df.drop('warehouse_non_work_day', axis=1)
    
    return df

def calculate_continuous_shut_down_before(df):
    df['warehouse_non_work_day'] = (~df['warehouse_work_day']).astype('int32')
    df = df.sort_values(by=['warehouse', 'date']).reset_index(drop=True)
    
    
    # Calculate continuous holidays shutdown before today --- for each warehouse. 
    # Helps Companies in staffing appropriately, Inventory mgmt etc
    
    df['continuous_shut_down_before'] = df.groupby('warehouse')['warehouse_non_work_day'].apply(
        lambda x: x[::-1].cumsum()[::-1].shift(-1, fill_value=0)
    )
    
    df = df.drop('warehouse_non_work_day', axis=1)
    LINEAR_COLS.append('continuous_shut_down_before')
    return df


def away_from_holiday(df, window=5, sort=True):
    df['holiday_(0)'] = False
    df.loc[~df['holiday_name'].isna(), 'holiday_(0)'] = True
    df['holiday_name'] = df['holiday_name'].fillna('not_holiday')
    BOOL_COLS.append('holiday_(0)')
    HOLIDAY_NAME_COLS.append('holiday_name')
    for i in range(1, window):
        df[[f'holiday_name_{i}', f'holiday_(-{i})']] = df.groupby('warehouse')[['holiday_name','holiday_(0)']].shift(-i)

        df[f'holiday_name_{i}'] = df[f'holiday_name_{i}'].fillna('not_holiday')
        df[f'holiday_(-{i})'] = df[f'holiday_(-{i})'].fillna(False)
        BOOL_COLS.append(f'holiday_(-{i})')
        HOLIDAY_NAME_COLS.append(f'holiday_name_{i}')
    return df
        
def after_holiday(df, window=1):
    for i in range(1, window + 1):
        shifted = df.groupby('warehouse')[['holiday_name', 'holiday_(0)']].shift(i)
        df[f'holiday_name_{-i}'] = shifted['holiday_name'].fillna('not_holiday')
        df[f'holiday_({i})'] = shifted['holiday_(0)'].fillna(False)
        
        BOOL_COLS.append(f'holiday_({i})')
        HOLIDAY_NAME_COLS.append(f'holiday_name_{-i}')
    
    return df

def add_season(num):
    if num in [3 ,4, 5]:
        return 1
    elif num in [6, 7, 8]:
        return 2
    elif num in [9, 10, 11]:
        return 3
    else:
        return 4

def split_date(df, date_column):
    date_type = 'datetime64[ns]'
    if df[date_column].dtype != date_type:
        df[date_column] = df[date_column].astype(date_type)
    
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['weekday'] = df[date_column].dt.weekday + 1
    df['season'] = df['month'].apply(add_season)
    
    LINEAR_COLS.append('season')
    
    return df

def get_advanced_date_information(df, date_column):
    df['week_of_year'] = df[date_column].dt.isocalendar().week.fillna(-1)
    df['quarter'] = df[date_column].dt.quarter.fillna(-1)
    df['is_month_start'] = df[date_column].dt.is_month_start.astype(int).fillna(-1)
    df['is_month_end'] = df[date_column].dt.is_month_end.astype(int).fillna(-1)
    df['is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int).fillna(-1)
    df['is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int).fillna(-1)
    df['is_weekend'] = df[date_column].dt.weekday.isin([5, 6]).astype(int)
    
    df['day_num'] = df[date_column].dt.dayofyear.fillna(-1)
    df['delta_days'] = (df[date_column] - df[date_column].min()).dt.days
    df['week_of_year_for_lag_order'] = df['day_num'] // 7 + 1
    
    return df

# test
def get_full_train_set(features, labels):
    return pd.merge(labels, features, how='left', on=PRIMARY_KEY)

def make_orders_ratio(features, labels):
    df = get_full_train_set(features, labels)
    
    orders_sum_of_week = df.groupby(['warehouse', 'week_of_year', 'year'])['orders'].sum().reset_index()
    df = pd.merge(df, orders_sum_of_week, how='left', on=['warehouse', 'week_of_year', 'year'], suffixes=('', '_sum_of_week'))
    df['orders_ratio_of_week'] = df['orders'] / df['orders_sum_of_week']
    df = df.drop('orders_sum_of_week', axis=1)
    
    orders_sum_of_month = df.groupby(['warehouse', 'month', 'year'])['orders'].sum().reset_index()
    df = pd.merge(df, orders_sum_of_month, how='left', on=['warehouse', 'month', 'year'], suffixes=('', '_sum_of_month'))
    df['orders_ratio_of_month'] = df['orders'] / df['orders_sum_of_month']
    df = df.drop('orders_sum_of_month', axis=1)
    
    labels = pd.merge(labels, df[['warehouse', 'date', 'orders_ratio_of_week', 'orders_ratio_of_month']], how='left', on=PRIMARY_KEY)
    return labels

def target_encoding(features, labels, cols, n_splits=5):
    train = features.loc[features['train'] == True]
    train = pd.merge(train, labels[target_cols+['date', 'warehouse']], how='left', on=['date', 'warehouse'])
    mean_target = train[target_cols].mean()
    for col in cols:
        for target in target_cols:
            features[f"{col}_{target}_target_encoded_mean"] = features[col].map(train.groupby(col)[target].mean()).fillna(0)

            features[f"{col}_{target}_target_encoded_std"] = features[col].map(train.groupby(col)[target].std()).fillna(0)
            na_count = features[f"{col}_{target}_target_encoded_mean"].isna().sum()
            if na_count > 0:
                print(f"{col} has nan count {na_count} before fillna, will fill {mean_target[target].mean()}")
            features[f"{col}_{target}_target_encoded_mean"] = features[f"{col}_{target}_target_encoded_mean"].fillna(mean_target[target].mean())
            na_count = features[f"{col}_{target}_target_encoded_mean"].isna().sum()
            if na_count > 0:
                print(f"{col} has nan count {na_count} after fillna")
    return features

def one_hot_encoding(fit:pd.DataFrame, transform:pd.DataFrame)->pd.DataFrame:
    enc = OneHotEncoder(sparse_output=False)
    enc.fit(fit.values)
    features = None
    for col in transform.columns:
        holiday_encoded = enc.transform(transform[[col]].values)
        encoded_df = pd.DataFrame(holiday_encoded, columns=enc.get_feature_names_out([col]))
        if features is None:
            features = encoded_df
        else:
            features = pd.concat([features, encoded_df], axis=1)
    return features

def one_hot_pipeline(df, cols):
    df_encoded = None
    for col in cols:
        if df_encoded is None:
            df_encoded = one_hot_encoding(df[[col]], df[[col]])
        else:
            df_encoded = pd.concat([df_encoded, one_hot_encoding(df[[col]], df[[col]])], axis=1)
    df = pd.concat([df, df_encoded], axis=1)
    drop_cols = list(set(cols) - set(PRIMARY_KEY))
    return df

def linear_encoding(features, cols):
    for col in cols:
        features[f"{col}_linear_encoded"] = features[col]
    return features

def binary_encode(features, cols):
    for col in cols:
        features[f'{col}_binary_encoded'] = features[col].astype('int64')
    return features