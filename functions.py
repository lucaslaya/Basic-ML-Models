import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error

#from sklearn import tree

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RANDOM_SEED=42
ohe = OneHotEncoder(categories='auto', handle_unknown='ignore')

def one_hot_encoding(train_df, test=False):
    # Create new OneHotEncoder
    #ohe = OneHotEncoder(categories='auto', handle_unknown='ignore')

    # All categorical features
    categorical_features = [col for col in train_df.columns if train_df[col].dtype == 'object']

    if test:
        feature_arr = ohe.transform(train_df[categorical_features]).toarray()
    else:
        feature_arr = ohe.fit_transform(train_df[categorical_features]).toarray()

    feature_labels = ohe.categories
    features = pd.DataFrame(feature_arr, columns=ohe.get_feature_names_out())

    train_df = pd.concat([train_df, features], axis=1).drop(columns=categorical_features, axis=1)

    return train_df

def train_tree_reg(X, y, min_samples_leaf=1, max_depth=100):
    # Initialize decision tree
    tree_reg = DecisionTreeRegressor(
        random_state=RANDOM_SEED,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth
    )

    tree_reg.fit(X, y)

    return tree_reg

def validate_tree(X, y, X_val, y_val, max_min_samples_leaf=50, max_max_depth=15):
    result_train_dict = {}
    result_val_dict = {}

    for m in range(3, max_max_depth+1, 2):
        for n in range(1, max_min_samples_leaf+1, 3):
            tree_reg = train_tree_reg(X, y, min_samples_leaf=n, max_depth=m)

            train_predicted = tree_reg.predict(X)
            val_predicted = tree_reg.predict(X_val)

            result_train_dict[m, n] = root_mean_squared_error(np.log(y), np.log(train_predicted))
            result_val_dict[m, n] = root_mean_squared_error(np.log(y_val), np.log(val_predicted))

    return result_train_dict, result_val_dict

def plot_errors(result_train_dict, result_val_dict, max_min_samples_leaf=50, max_max_depth=15):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
    colors= ['black', 'red', 'blue', 'green', 'orange', 'yellow', 'gray']
    color_index = 0
    custom_lines = np.array([])
    custom_names = np.array([])

    x_axis = np.arange(start=1, stop=max_min_samples_leaf+1, step=3)

    for m in range(3, max_max_depth+1, 2):

        line_plot_train = []
        line_plot_val = []

        for n in x_axis:
            line_plot_train.append(result_train_dict.get((m,n)))
            line_plot_val.append(result_val_dict.get((m,n)))

        ax1.plot(x_axis, line_plot_train, alpha=0.5, c=colors[color_index], linestyle='--')
        ax2.plot(x_axis, line_plot_val, alpha=0.5, c=colors[color_index])

        color_line = Line2D([0], [0], color=colors[color_index], lw=4)
        custom_lines = np.append(custom_lines,color_line)
        color_name = 'max_depth =' + str(m)
        custom_names = np.append(custom_names,color_name)
        color_index+=1

    ax1.set_xlim(max_min_samples_leaf, 0)
    ax2.set_xlim(max_min_samples_leaf, 0)

    plt.xticks(np.arange(start=1, stop=max_min_samples_leaf, step=2), rotation=90, size=10)

    ax1.grid(True)
    ax2.grid(True)

    plt.xlabel('# min_samples_leaf')
    plt.ylabel('RMSE')

    ax1.legend(custom_lines, custom_names)

    plt.show()

def check_validation(X, y, X_train, y_train, X_val, y_val, min_samples_leaf, max_depth):
    tree_reg = DecisionTreeRegressor(random_state=RANDOM_SEED, min_samples_leaf=min_samples_leaf, max_depth=max_depth)
    tree_reg.fit(X_train, y_train)

    train_predicted = tree_reg.predict(X_train)
    val_predicted = tree_reg.predict(X_val)

    print(f'y_train validated result: {root_mean_squared_error(np.log(y_train), np.log(train_predicted))}')
    print(f'y_val validated result: {root_mean_squared_error(np.log(y_val), np.log(val_predicted))}')

    tree_reg = DecisionTreeRegressor(
        random_state=RANDOM_SEED,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth
    )
    tree_reg.fit(X, y)

    train_predicted = tree_reg.predict(X)

    print(f'y validated result: {root_mean_squared_error(np.log(y), np.log(train_predicted))}')

    return tree_reg
