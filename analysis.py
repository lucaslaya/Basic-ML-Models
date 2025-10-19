import pandas as pd
import numpy as np

from functions import *

MIN_SAMPLES_LEAF = 16
MAX_DEPTH = 11

def create_model(train, test):
    train_df = one_hot_encoding(train)

    X = train_df.drop(columns='SalePrice').fillna(0)
    y = train_df['SalePrice'].values
    #print(f'X Shape: {X_train.shape}, y Shape: {y_train.shape}')

    X_train, X_val, y_train, y_val = train_test_split(X,
                                                      y,
                                                      test_size=0.3,
                                                      random_state=RANDOM_SEED)

    result_train_dict, result_val_dict = validate_tree(X_train, y_train, X_val, y_val)

    plot_errors(result_train_dict, result_val_dict)

    tree_reg = check_validation(X, y, X_train, y_train, X_val, y_val, MIN_SAMPLES_LEAF, MAX_DEPTH)

    test_df = one_hot_encoding(test, test=True)

    X_test = test_df.drop(columns='SalePrice').fillna(0)
    y_test = test_df[['SalePrice']].copy()

    y_test['SalePrice_predicted'] = tree_reg.predict(X_test)
    print(y_test.head())

    print(f'Test set error: {root_mean_squared_error(np.log(y_test['SalePrice']), np.log(y_test['SalePrice_predicted']))}')

def features1(df):
    """Select optimal columns and compute TotalSF"""
    # Selected columns for optimal model
    selected_cols = [
        # Quality & Condition
        'OverallQual', 'OverallCond', 'ExterQual', 'KitchenQual',
        # Size/Area Features
        'GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea', 'LotArea',
        # Age Variables
        'YearBuilt', 'YearRemodAdd', 'GarageYrBlt',
        # Location
        'Neighborhood', 'MSZoning',
        # Room Counts
        'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars',
        # Basement Features
        'BsmtQual', 'BsmtCond', 'BsmtFinSF1', 'BsmtFinType1',
        # Other Key Features
        'CentralAir', 'Fireplaces', 'FireplaceQu', 'Foundation',
        'GarageType', 'GarageFinish', 'PavedDrive'
    ]

    # Add SalePrice if it exists (for training data)
    if 'SalePrice' in df.columns:
        selected_cols.append('SalePrice')

    # Select only columns that exist in the dataframe
    available_cols = [col for col in selected_cols if col in df.columns]
    df_filtered = df[available_cols].copy()

    # Compute TotalSF
    df_filtered['TotalSF'] = df_filtered['TotalBsmtSF'].fillna(0) + df_filtered['1stFlrSF'].fillna(0) + df_filtered['2ndFlrSF'].fillna(0)

    #print(df_filtered.head())
    return df_filtered

if __name__ == "__main__":
    # Read data
    train_df = pd.read_csv('AmesHousingData/Ames_train.csv')
    test_df = pd.read_csv('AmesHousingData/Ames_test.csv')

    print(f'Train shape: {train_df.shape} \nTest shape: {test_df.shape} \nSale Price Mean: {int(train_df.SalePrice.mean())}')

    train_df = features1(train_df)
    test_df = features1(test_df)

    create_model(train_df, test_df)
    #run_model_all_combinations(train_df, test_df)


