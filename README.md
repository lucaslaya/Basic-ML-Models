# Machine Learning Models

## Feature Engineering
Drop:
- Alley

Weird:
- LotShape
- LotConfig
- Condition 1 + 2 in one variable
- Exterior 1st + 2nd + MasVnrType + Foundation

### Unchanged Variables
- Street: convert to 0 and 1 binary instead of strings
- OverallQual
- OverallCond
- ExterQual: convert to numerical
- ExterCond: convert to numerical
- BsmtCond: convert to numerical
- BsmtExposure: convert to numerical
  - CONTINUE FROM HERE ON FEATURE DESCRIPTION LIST

### OHE Variables
- LotShape (Weird)
- Utilities
- LotConfig (Weird)
- LandSlope
- LandContour
- BldgType
- HouseStyle
- RoofStyle
- Roof Matl


### Variables that I could Encode (Categorical to numerical)
- MSSubClass
- MSZoning
- Neighborhood
- Condition 1 + 2 in one variable
- Exterior 1st + 2nd + MasVnrType + Foundation

### Compute variables
- Age: YearBuilt - current year
- Year since remod: YearRemodAdd - current year
- BsmtQual - Convert to height values



