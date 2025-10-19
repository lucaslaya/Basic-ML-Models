import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def analyze_features(train_df):
    print("=== AMES HOUSING FEATURE ANALYSIS ===\n")

    # Basic info
    print(f"Dataset shape: {train_df.shape}")
    print(f"Target variable (SalePrice) stats:")
    print(f"  Mean: ${train_df['SalePrice'].mean():,.0f}")
    print(f"  Median: ${train_df['SalePrice'].median():,.0f}")
    print(f"  Std: ${train_df['SalePrice'].std():,.0f}")
    print(f"  Range: ${train_df['SalePrice'].min():,.0f} - ${train_df['SalePrice'].max():,.0f}\n")

    # Correlation analysis for numerical features
    numerical_features = train_df.select_dtypes(include=[np.number]).columns
    numerical_features = [col for col in numerical_features if col != 'SalePrice']

    correlations = []
    for feature in numerical_features:
        if train_df[feature].nunique() > 1:  # Skip constant features
            corr, _ = pearsonr(train_df[feature].fillna(0), train_df['SalePrice'])
            correlations.append((feature, abs(corr), corr))

    correlations.sort(key=lambda x: x[1], reverse=True)

    print("=== TOP 20 NUMERICAL FEATURES BY CORRELATION ===")
    for i, (feature, abs_corr, corr) in enumerate(correlations[:20]):
        print(f"{i+1:2d}. {feature:<20} | Correlation: {corr:6.3f} | Abs: {abs_corr:.3f}")

    # Analyze categorical features
    categorical_features = train_df.select_dtypes(include=['object']).columns

    print(f"\n=== CATEGORICAL FEATURES ANALYSIS ===")
    categorical_importance = []

    for feature in categorical_features:
        if train_df[feature].nunique() > 1 and train_df[feature].nunique() < 50:
            # Calculate mean price by category
            group_means = train_df.groupby(feature)['SalePrice'].agg(['mean', 'count'])
            price_range = group_means['mean'].max() - group_means['mean'].min()
            categorical_importance.append((feature, price_range, train_df[feature].nunique()))

    categorical_importance.sort(key=lambda x: x[1], reverse=True)

    print("Top categorical features by price range:")
    for i, (feature, price_range, unique_count) in enumerate(categorical_importance[:15]):
        print(f"{i+1:2d}. {feature:<20} | Price Range: ${price_range:8.0f} | Categories: {unique_count}")

    return correlations, categorical_importance

def design_computed_variables(train_df):
    print(f"\n=== DESIGNING COMPUTED VARIABLES ===")

    # Create computed variables
    computed_df = train_df.copy()

    # 1. Total Square Footage
    computed_df['Total_SF'] = (computed_df['Total Bsmt SF'].fillna(0) +
                              computed_df['1st Flr SF'] +
                              computed_df['2nd Flr SF'].fillna(0))

    # 2. Age of house
    computed_df['House_Age'] = computed_df['Yr Sold'] - computed_df['Year Built']

    # 3. Years since remodel
    computed_df['Years_Since_Remod'] = computed_df['Yr Sold'] - computed_df['Year Remod/Add']

    # 4. Total bathrooms
    computed_df['Total_Bathrooms'] = (computed_df['Full Bath'] +
                                     computed_df['Half Bath'] * 0.5 +
                                     computed_df['Bsmt Full Bath'].fillna(0) +
                                     computed_df['Bsmt Half Bath'].fillna(0) * 0.5)

    # 5. Total porch area
    computed_df['Total_Porch_SF'] = (computed_df['Open Porch SF'].fillna(0) +
                                    computed_df['Enclosed Porch'].fillna(0) +
                                    computed_df['3Ssn Porch'].fillna(0) +
                                    computed_df['Screen Porch'].fillna(0))

    # 6. Price per square foot (for analysis)
    computed_df['Price_per_SF'] = computed_df['SalePrice'] / computed_df['Total_SF']

    # 7. Garage to lot ratio
    computed_df['Garage_to_Lot_Ratio'] = computed_df['Garage Area'].fillna(0) / computed_df['Lot Area']

    # 8. Overall quality-condition interaction
    computed_df['Quality_Condition_Score'] = computed_df['Overall Qual'] * computed_df['Overall Cond']

    # 9. Basement quality score
    basement_quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    computed_df['Bsmt_Quality_Score'] = computed_df['Bsmt Qual'].map(basement_quality_map).fillna(0)

    # 10. Kitchen quality score
    kitchen_quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    computed_df['Kitchen_Quality_Score'] = computed_df['Kitchen Qual'].map(kitchen_quality_map).fillna(3)

    # Test computed variables
    computed_vars = ['Total_SF', 'House_Age', 'Years_Since_Remod', 'Total_Bathrooms',
                    'Total_Porch_SF', 'Garage_to_Lot_Ratio', 'Quality_Condition_Score',
                    'Bsmt_Quality_Score', 'Kitchen_Quality_Score']

    print("Correlations of computed variables:")
    for var in computed_vars:
        if var in computed_df.columns:
            corr = computed_df[var].corr(computed_df['SalePrice'])
            print(f"  {var:<25} | Correlation: {corr:6.3f}")

    return computed_df, computed_vars

if __name__ == "__main__":
    train_df = pd.read_csv('AmesHousingData/Ames_train.csv')

    correlations, categorical_importance = analyze_features(train_df)
    computed_df, computed_vars = design_computed_variables(train_df)
