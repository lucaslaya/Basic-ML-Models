import pandas as pd
from feature_engineering import FeatureEngineer

# Load data
train_df = pd.read_csv('AmesHousingData/Ames_train.csv')

# Apply feature engineering
engineer = FeatureEngineer(reference_year=2025)
train_df = engineer.apply_all_conversions(train_df)

# Find all object (string) columns
categorical_cols = [col for col in train_df.columns if train_df[col].dtype == 'object']

print('Categorical columns remaining after feature engineering:')
for col in sorted(categorical_cols):
    print(f'  {col}')
print(f'\nTotal: {len(categorical_cols)} categorical columns')
