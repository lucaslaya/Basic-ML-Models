import pandas as pd

from Models.decision_tree import DecisionTree
from Models.random_forest import RandomForest

from feature_engineering import FeatureEngineer 

def decision_tree(train_set, test_set, categorical_features=[]):
    model = DecisionTree(train_set, test_set, categorical_features)

    model.prepare_train_set()
    model.fit_and_validate()
    #model.plot_validation()
    model.print_validation()

    max_depth = int(input("max_depth: "))
    min_samples_leaf = int(input("min_samples_leaf: "))
    model.check_validation(max_depth=max_depth, min_samples_leaf=min_samples_leaf)

    model.test_model()

def random_forest(train_set, test_set, categorical_features=[]):
    model = RandomForest(train_set, test_set, categorical_features)

    model.prepare_train_set()
    model.fit_and_validate()
    #model.plot_validation()
    model.print_validation()

    n_estimators = int(input("n_estimators: "))
    max_depth = int(input("max_depth: "))
    max_features = float(input("max_features: "))
    max_samples = float(input("max_samples: "))
    model.check_validation(n_estimators, max_depth, max_features, max_samples)

    model.test_model()

def random_forest_optimized(train_set, test_set, categorical_features=[]):
    """
    Optimized Random Forest using GridSearchCV for parallel processing.
    Much faster than the regular random_forest() function.
    """
    model = RandomForest(train_set, test_set, categorical_features)

    model.prepare_train_set()
    model.fit_and_validate_optimized()  # Uses GridSearchCV with parallel processing
    #model.plot_validation()
    model.print_validation()

    n_estimators = int(input("n_estimators: "))
    max_depth = int(input("max_depth: "))
    max_features = float(input("max_features: "))
    max_samples = float(input("max_samples: "))
    model.check_validation(n_estimators, max_depth, max_features, max_samples)

    model.test_model()

def feature_engineering1(df):
    # One Hot Encoded Variables
    categorical_features = [
        'LotShape', 'Utilities', 'LotConfig', 'LandSlope', 'LandContour',
        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Heating',
        'Electrical', 'Functional', 'GarageType', 'PavedDrive', 'MiscFeature',
        'SaleType', 'SaleCondition', 'MSZoning', 'Neighborhood', 'Condition1',
        'Condition2', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
        'BsmtFinType1', 'BsmtFinType2', 'GarageFinish'
    ]

    # Dropped columns
    df.drop(columns=['Alley', 'PID'])



if __name__ == "__main__":
    train_df = pd.read_csv('AmesHousingData/Ames_train.csv')
    test_df = pd.read_csv('AmesHousingData/Ames_test.csv')

    engineer = FeatureEngineer(reference_year=2025)

    categorical_features = [
        'LotShape', 'Utilities', 'LotConfig', 'LandSlope', 'LandContour',
        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Heating',
        'Electrical', 'Functional', 'GarageType', 'PavedDrive', 'MiscFeature',
        'SaleType', 'SaleCondition', 'MSZoning', 'Neighborhood', 'Condition1',
        'Condition2', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
        'BsmtFinType1', 'BsmtFinType2', 'GarageFinish'
    ]

    train_df = engineer.apply_all_conversions(train_df)
    test_df = engineer.apply_all_conversions(test_df)

    decision_tree(train_df, test_df, categorical_features)
    #random_forest()
    #random_forest_optimized(train_df, test_df, categorical_features)
