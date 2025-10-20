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
- TotalBsmtSF
- BsmtUnfSF
- HeatingQC: convert to numerical
- CentralAir: Boolean
- 1stFlrSF
- 2ndFlrSF
- LowQualFinSF
- GrLivArea
- BsmtFullBathmachine
- BsmtHalfBath
- FullBath
- HalfBath
- Bedroom
- Kitchen
- KitchenQual: convert to numerical
- TotRmsAbvGrd 
- Fireplaces
- FireplaceQu: convert to numerical
- GarageCars 
- GarageArea 
- GarageQual: convert to numerical
- GarageCond: convert to numerical
- WoodDeckSF
- OpenPorchSF
- EnclosedPorch
- 3SsnPorch
- ScreenPorch
- PoolArea
- PoolQc: convert to numerical
- Fence: convert to numerical
- MiscValue 
- MoSold
- YrSold


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
- Heating
- Electrical
- Functional
- GarageTyp
- PavedDrive
- MiscFeature
- SaleType
- SaleCondition

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
- GarageAge: GarageYrBlt - current year



