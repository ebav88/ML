# Constants
PRIMARY_KEY = ['warehouse', 'date']
CUT_DATE = '2023-12-15'
DROP_COLS = ['id', 'warehouse_limited']
LINEAR_COLS = ['year', 'day_num', 'continuous_holiday', 'continuous_shut_down_before', 'delta_days']
CATE_COLS = ['warehouse']
DATE_COLS = ['month', 'week_of_year', 'quarter']
WEEKDAY_COLS = ['weekday']
BOOL_COLS = ['holiday', 'shops_closed', 'school_holidays', 'winter_school_holidays', 
             'is_quarter_start', 'is_quarter_end', 'is_month_start', 'is_month_end',
             'is_weekend']
HOLIDAY_NAME_COLS = []
target_cols = ['orders', 'user_activity_1', 'user_activity_2']
FEATURES_COLS = [DROP_COLS, LINEAR_COLS, CATE_COLS, DATE_COLS, WEEKDAY_COLS, BOOL_COLS, HOLIDAY_NAME_COLS]

LABEL1 = ['shutdown', 'snow', 'precipitation', 'mov_change', 'frankfurt_shutdown', 'blackout', 'mini_shutdown', 'user_activity_1', 'user_activity_2']
LABEL2 = ['orders']
