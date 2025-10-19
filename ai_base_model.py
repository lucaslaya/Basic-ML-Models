# -*- coding: utf-8 -*-
"""
BaseModel class for machine learning models
Contains shared functionality across Decision Tree, Random Forest, and Gradient Boosting models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class BaseModel:
    """
    Base class for machine learning models with common preprocessing,
    feature engineering, and evaluation methods.
    """

    def __init__(self, random_state=42):
        """
        Initialize the BaseModel

        Parameters:
        -----------
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.ohe = None
        self.encoding_lookup = None
        self.train_df = None
        self.test_df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.model = None

    def read_data(self, train_path, test_path=None, features=None):
        """
        Read train and test data from csv files

        Parameters:
        -----------
        train_path : str
            Path to training csv file
        test_path : str, optional
            Path to test csv file
        features : list, optional
            List of feature names to keep. If None, keeps all columns.
        """
        self.train_df = pd.read_csv(train_path)

        if features is not None:
            self.train_df = self.train_df[features]

        if test_path is not None:
            self.test_df = pd.read_csv(test_path)
            if features is not None:
                self.test_df = self.test_df[features]

        return self.train_df

    def apply_one_hot_encoding(self, categorical_features, df=None):
        """
        Apply One Hot Encoding to categorical features

        Parameters:
        -----------
        categorical_features : list
            List of categorical feature names to encode
        df : DataFrame, optional
            DataFrame to encode. If None, uses self.train_df

        Returns:
        --------
        DataFrame
            DataFrame with one-hot encoded features
        """
        if df is None:
            df = self.train_df

        # We define our One Hot Encoder
        if self.ohe is None:
            self.ohe = OneHotEncoder(categories='auto', handle_unknown='ignore')

            # We apply the encoding to our train dataset
            feature_arr = self.ohe.fit_transform(df[categorical_features]).toarray()
        else:
            # For test set, we just APPLY the One Hot Encoder
            # So we only use "transform", not "fit" and "transform"
            feature_arr = self.ohe.transform(df[categorical_features]).toarray()

        # We also store the names of the categories for the new column names
        feature_labels = self.ohe.categories_

        # We apply the new feature names
        features = pd.DataFrame(feature_arr, columns=self.ohe.get_feature_names_out())

        # And then combine with the rest of the numerical variables
        df_encoded = pd.concat([df, features], axis=1).drop(columns=categorical_features, axis=1)

        return df_encoded

    def prepare_features(self, target_col='SalePrice', fillna_value=0):
        """
        Prepare X (features) and y (target) from training data

        Parameters:
        -----------
        target_col : str, default='SalePrice'
            Name of the target column
        fillna_value : float, default=0
            Value to use for filling NaN values

        Returns:
        --------
        tuple
            (X, y) DataFrames
        """
        self.X = self.train_df.drop(columns=[target_col]).fillna(fillna_value)
        self.y = self.train_df[[target_col]]

        return self.X, self.y

    def engineer_features(self, df):
        """
        Create engineered features from existing features

        Parameters:
        -----------
        df : DataFrame
            DataFrame to add engineered features to

        Returns:
        --------
        DataFrame
            DataFrame with engineered features added
        """
        # Feature Engineering:
        df['Garage_area_ratio'] = df['Garage Area'] / df['Gr Liv Area']
        df['Total_area'] = df['Garage Area'] + df['Gr Liv Area']

        # Engineering more variables by applying non-linear
        # transformations to original variables
        for column in ['Gr Liv Area',
                       'Garage Area',
                       'Year Built',
                       'Garage_area_ratio',
                       'Total_area']:
            df[column + '_log'] = np.log(df[column] + 0.0001)

        return df

    def create_train_val_split(self, X, y, test_size=0.3, random_state=None):
        """
        Create train/validation split

        Parameters:
        -----------
        X : DataFrame
            Features
        y : DataFrame
            Target
        test_size : float, default=0.3
            Proportion of dataset to include in validation split
        random_state : int, optional
            Random seed. If None, uses self.random_state

        Returns:
        --------
        tuple
            (X_train, X_val, y_train, y_val)
        """
        if random_state is None:
            random_state = self.random_state

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        return self.X_train, self.X_val, self.y_train, self.y_val

    def calculate_rmse(self, y_true, y_pred, use_log=True, squared=False):
        """
        Calculate RMSE (Root Mean Squared Error)

        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        use_log : bool, default=True
            Whether to apply log transformation before calculating RMSE
        squared : bool, default=False
            If True, returns MSE instead of RMSE

        Returns:
        --------
        float
            RMSE value
        """
        if use_log:
            return mean_squared_error(
                np.log(y_true),
                np.log(y_pred),
                squared=squared
            )
        else:
            return mean_squared_error(y_true, y_pred, squared=squared)

    def fit(self, X, y):
        """
        Fit the model. To be implemented by child classes.

        Parameters:
        -----------
        X : DataFrame
            Features
        y : DataFrame or Series
            Target
        """
        raise NotImplementedError("Child classes must implement fit method")

    def predict(self, X):
        """
        Make predictions using the trained model

        Parameters:
        -----------
        X : DataFrame
            Features to predict on

        Returns:
        --------
        array
            Predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        return self.model.predict(X)

    def evaluate(self, X, y, use_log=True):
        """
        Evaluate model performance

        Parameters:
        -----------
        X : DataFrame
            Features
        y : DataFrame or Series
            True target values
        use_log : bool, default=True
            Whether to use log transformation for RMSE calculation

        Returns:
        --------
        float
            RMSE score
        """
        predictions = self.predict(X)
        return self.calculate_rmse(y, predictions, use_log=use_log)

    def plot_feature_importance(self, feature_names):
        """
        Plot feature importances

        Parameters:
        -----------
        feature_names : array-like
            Names of features
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")

        feat_importance = self.model.feature_importances_
        tree_importances = pd.Series(feat_importance, index=feature_names).sort_values(ascending=False)

        fig, ax = plt.subplots()
        tree_importances.plot.bar(ax=ax)
        ax.set_title("Feature importances")
        ax.set_ylabel("Mean decrease in impurity (normalized)")
        fig.tight_layout()

        return fig, ax

    def prepare_test_data(self, target_col='SalePrice', fillna_value=0):
        """
        Prepare test data for predictions

        Parameters:
        -----------
        target_col : str, default='SalePrice'
            Name of the target column
        fillna_value : float, default=0
            Value to use for filling NaN values

        Returns:
        --------
        tuple
            (X_test, y_test) DataFrames
        """
        if self.test_df is None:
            raise ValueError("Test data has not been loaded. Call read_data() with test_path.")

        self.X_test = self.test_df.drop(columns=[target_col]).fillna(fillna_value)
        self.y_test = self.test_df[[target_col]].copy()

        return self.X_test, self.y_test
