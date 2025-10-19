import pandas as pd
import time

from Models.decision_tree import DecisionTree
from Models.random_forest import RandomForest

def decision_tree():
    model = DecisionTree(
    pd.read_csv('AmesHousingData/Ames_train.csv'), 
    pd.read_csv('AmesHousingData/Ames_test.csv')
    )

    model.prepare_train_set()
    model.fit_and_validate()
    model.print_validation()

    max_depth = int(input("max_depth: "))
    min_samples_leaf = int(input("min_samples_leaf: "))
    model.check_validation(max_depth=max_depth, min_samples_leaf=min_samples_leaf)

    model.test_model()

def random_forest():
    model = RandomForest(
    pd.read_csv('AmesHousingData/Ames_train.csv'), 
    pd.read_csv('AmesHousingData/Ames_test.csv')
    )

    model.prepare_train_set()
    model.fit_and_validate()
    model.plot_validation()
    model.print_validation()

    n_estimators = int(input("n_estimators: "))
    max_depth = int(input("max_depth: "))
    max_features = int(input("max_features: "))
    max_samples = int(input("max_samples: "))
    model.check_validation(n_estimators, max_depth, max_features, max_samples)


# decision_tree()
random_forest()
