"""
Feature Engineering Module for Ames Housing Dataset
Converts columns as specified in README.md
"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    Feature engineering class for Ames Housing dataset.
    Handles conversion of categorical to numerical, binary, and computed features.
    """

    # Quality mappings (Ex > Gd > TA > Fa > Po > NA)
    QUALITY_MAP_5 = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.nan: 0}

    # Exposure mapping (Gd > Av > Mn > No > NA)
    EXPOSURE_MAP = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, np.nan: 0}

    # Fence mapping (GdPrv > MnPrv > GdWo > MnWw > NA)
    FENCE_MAP = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, np.nan: 0}

    # Basement Quality to height mapping (in inches)
    BSMT_QUAL_HEIGHT_MAP = {
        'Ex': 100,  # Excellent (100+ inches)
        'Gd': 90,   # Good (90-99 inches)
        'TA': 80,   # Typical (80-89 inches)
        'Fa': 70,   # Fair (70-79 inches)
        'Po': 60,   # Poor (<70 inches)
        np.nan: 0   # No basement
    }

    def __init__(self, reference_year=2025):
        """
        Initialize the FeatureEngineer.

        Parameters:
        -----------
        reference_year : int
            Year to use for age calculations (default: 2025)
        """
        self.reference_year = reference_year

    def convert_to_binary(self, df, column, positive_value):
        """
        Convert a column to binary (0 or 1).

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        column : str
            Column name to convert
        positive_value : str
            Value that should be converted to 1 (others become 0)

        Returns:
        --------
        pd.Series
            Binary series
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")

        return (df[column] == positive_value).astype(int)

    def convert_to_quality_numerical(self, df, column, mapping=None):
        """
        Convert quality columns to numerical values.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        column : str
            Column name to convert
        mapping : dict, optional
            Custom mapping dictionary. If None, uses default QUALITY_MAP_5

        Returns:
        --------
        pd.Series
            Numerical series
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")

        if mapping is None:
            mapping = self.QUALITY_MAP_5

        return df[column].map(mapping).fillna(0)

    def compute_age(self, df, year_column):
        """
        Compute age based on year column.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        year_column : str
            Column containing year values

        Returns:
        --------
        pd.Series
            Age values
        """
        if year_column not in df.columns:
            raise ValueError(f"Column '{year_column}' not found in dataframe")

        return self.reference_year - df[year_column]

    def apply_all_conversions(self, df, inplace=False):
        """
        Apply all feature engineering transformations as specified in README.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        inplace : bool
            Whether to modify the dataframe in place

        Returns:
        --------
        pd.DataFrame
            Transformed dataframe
        """
        if not inplace:
            df = df.copy()

        # Binary conversions
        if 'Street' in df.columns:
            df['Street'] = self.convert_to_binary(df, 'Street', 'Pave')

        if 'CentralAir' in df.columns:
            df['CentralAir'] = self.convert_to_binary(df, 'CentralAir', 'Y')

        # Quality conversions (using standard 5-level quality mapping)
        quality_columns = ['ExterQual', 'ExterCond', 'BsmtCond', 'HeatingQC',
                          'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond',
                          'PoolQC']

        for col in quality_columns:
            if col in df.columns:
                df[col] = self.convert_to_quality_numerical(df, col)

        # Basement Exposure
        if 'BsmtExposure' in df.columns:
            df['BsmtExposure'] = self.convert_to_quality_numerical(
                df, 'BsmtExposure', mapping=self.EXPOSURE_MAP
            )

        # Fence
        if 'Fence' in df.columns:
            df['Fence'] = self.convert_to_quality_numerical(
                df, 'Fence', mapping=self.FENCE_MAP
            )

        # Computed variables - Age
        if 'YearBuilt' in df.columns:
            df['Age'] = self.compute_age(df, 'YearBuilt')

        # Year since remodel
        if 'YearRemod/Add' in df.columns:
            df['YearsSinceRemod'] = self.compute_age(df, 'YearRemod/Add')

        # Basement Quality to Height
        if 'BsmtQual' in df.columns:
            df['BsmtQual'] = self.convert_to_quality_numerical(
                df, 'BsmtQual', mapping=self.BSMT_QUAL_HEIGHT_MAP
            )

        # Garage Age
        if 'GarageYrBlt' in df.columns:
            df['GarageAge'] = self.compute_age(df, 'GarageYrBlt')
            # Handle cases where garage doesn't exist (NaN year)
            df['GarageAge'] = df['GarageAge'].fillna(0)

        # Drop Alley as specified
        if 'Alley' in df.columns:
            df = df.drop(columns=['Alley'])

        return df


# Convenience functions for easy use
def convert_column_to_binary(df, column, positive_value, inplace=False):
    """
    Standalone function to convert a single column to binary.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to convert
    positive_value : str
        Value that should be converted to 1
    inplace : bool
        Whether to modify the dataframe in place

    Returns:
    --------
    pd.DataFrame or None
        Modified dataframe if inplace=False, None otherwise
    """
    engineer = FeatureEngineer()

    if inplace:
        df[column] = engineer.convert_to_binary(df, column, positive_value)
        return None
    else:
        df_copy = df.copy()
        df_copy[column] = engineer.convert_to_binary(df_copy, column, positive_value)
        return df_copy


def convert_column_to_quality(df, column, mapping=None, inplace=False):
    """
    Standalone function to convert a single quality column to numerical.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to convert
    mapping : dict, optional
        Custom mapping dictionary
    inplace : bool
        Whether to modify the dataframe in place

    Returns:
    --------
    pd.DataFrame or None
        Modified dataframe if inplace=False, None otherwise
    """
    engineer = FeatureEngineer()

    if inplace:
        df[column] = engineer.convert_to_quality_numerical(df, column, mapping)
        return None
    else:
        df_copy = df.copy()
        df_copy[column] = engineer.convert_to_quality_numerical(df_copy, column, mapping)
        return df_copy


# Example usage
if __name__ == "__main__":
    # Example: Load and transform data
    train_df = pd.read_csv('AmesHousingData/Ames_train.csv')

    # Method 1: Use the class for all transformations
    engineer = FeatureEngineer(reference_year=2025)
    train_transformed = engineer.apply_all_conversions(train_df, inplace=False)

    print("Transformations applied successfully!")
    print(f"Original shape: {train_df.shape}")
    print(f"Transformed shape: {train_transformed.shape}")

    # Method 2: Use individual functions for specific columns
    # df = convert_column_to_binary(df, 'Street', 'Pave', inplace=False)
    # df = convert_column_to_quality(df, 'ExterQual', inplace=False)
