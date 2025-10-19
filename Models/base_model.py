import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

class BaseModel:
    def __init__(self, train_df, test_df, categorical_features=[], random_seed=42):
        self.random_seed = random_seed
        self.ohe = OneHotEncoder(categories='auto', handle_unknown='ignore')
        self.categorical_features = categorical_features

        if categorical_features == []:
            self.categorical_features = [col for col in train_df.columns if train_df[col].dtype == 'object']

        self.train_df = self.one_hot_encoding(train_df, test=False)
        self.test_df = self.one_hot_encoding(test_df, test=True)

        # Training data
        self.X = None
        self.y = None

        self.X_train = None
        self.y_train = None

        self.X_val = None
        self.y_val = None

        # Test data
        self.X_test = None
        self.y_test = None

        self.model = None


    def one_hot_encoding(self, df, test):
        # All categorical features
        categorical_features = self.categorical_features

        if test:
            feature_arr = self.ohe.transform(df[categorical_features]).toarray()
            print("One hot encoding complete on TEST DATA")
        else:
            feature_arr = self.ohe.fit_transform(df[categorical_features]).toarray()
            print("One hot encoding complete on TRAIN DATA")

        feature_labels = self.ohe.categories_
        features = pd.DataFrame(feature_arr, columns=self.ohe.get_feature_names_out())

        encoded_df = pd.concat([df, features], axis=1).drop(columns=categorical_features, axis=1)
        return encoded_df

    def prepare_train_set(self):
        self.X = self.train_df.drop(columns=['SalePrice']).fillna(0)
        self.y = self.train_df['SalePrice'].values

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y,
            test_size=0.2,
            random_state=self.random_seed
        )

        return self.X, self.y, self.X_train, self.X_val, self.y_train, self.y_val

    def fit_and_validate(self):
        """
        Fit models and check parameters to develop BaseModel
        """
        raise NotImplementedError("Child classes must implement fit_and_validate method")

    def plot_validation(self):
        """
        Plot validation results
        """
        raise NotImplementedError("Child classes must implement plot_validation method")

    def check_validation(self):
        raise NotImplementedError("Child classes must implement check_validation method")

    def test_model(self, use_log=True):
        self.X_test = self.test_df.drop(columns='SalePrice').fillna(0)
        self.y_test = self.test_df[['SalePrice']].copy()

        predictions = self.model.predict(self.X_test)

        result = self.calculate_rmse(self.y_test, predictions, use_log=use_log)
        print('------------------------------------------------')
        print(f'test validated result: {result}')
        print('------------------------------------------------')


        return result

    def calculate_rmse(self, y, y_predicted, use_log=True):
        if use_log:
            return root_mean_squared_error(np.log(y), np.log(y_predicted))
        else:
            return root_mean_squared_error(y, y_predicted)

