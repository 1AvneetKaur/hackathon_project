# Hackathon Project Test

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#IMPORTING THE DATASET
test_set = pd.read_csv('test.csv')

# FILL MISSING VALUES
test_set['LotFrontage']=test_set['LotFrontage'].fillna(test_set['LotFrontage'].mean())
test_set['MSZoning']=test_set['MSZoning'].fillna(test_set['MSZoning'].mode()[0])
test_set.drop(['Alley'],axis=1,inplace=True)
test_set['BsmtCond']=test_set['BsmtCond'].fillna(test_set['BsmtCond'].mode()[0])
test_set['BsmtQual']=test_set['BsmtQual'].fillna(test_set['BsmtQual'].mode()[0])
test_set['FireplaceQu']=test_set['FireplaceQu'].fillna(test_set['FireplaceQu'].mode()[0])
test_set['GarageType']=test_set['GarageType'].fillna(test_set['GarageType'].mode()[0])
test_set.drop(['GarageYrBlt'],axis=1,inplace=True)
test_set['GarageFinish']=test_set['GarageFinish'].fillna(test_set['GarageFinish'].mode()[0])
test_set['GarageQual']=test_set['GarageQual'].fillna(test_set['GarageQual'].mode()[0])
test_set['GarageCond']=test_set['GarageCond'].fillna(test_set['GarageCond'].mode()[0])
test_set.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
test_set.drop(['Id'],axis=1,inplace=True)
test_set['MasVnrType']=test_set['MasVnrType'].fillna(test_set['MasVnrType'].mode()[0])
test_set['MasVnrArea']=test_set['MasVnrArea'].fillna(test_set['MasVnrArea'].mode()[0])
test_set['BsmtExposure']=test_set['BsmtExposure'].fillna(test_set['BsmtExposure'].mode()[0])
test_set['BsmtFinType2']=test_set['BsmtFinType2'].fillna(test_set['BsmtFinType2'].mode()[0])
test_set['Utilities']=test_set['Utilities'].fillna(test_set['Utilities'].mode()[0])
test_set['Exterior1st']=test_set['Exterior1st'].fillna(test_set['Exterior1st'].mode()[0])
test_set['Exterior2nd']=test_set['Exterior2nd'].fillna(test_set['Exterior2nd'].mode()[0])
test_set['BsmtFinType1']=test_set['BsmtFinType1'].fillna(test_set['BsmtFinType1'].mode()[0])
test_set['BsmtFinSF1']=test_set['BsmtFinSF1'].fillna(test_set['BsmtFinSF1'].mean())
test_set['BsmtFinSF2']=test_set['BsmtFinSF2'].fillna(test_set['BsmtFinSF2'].mean())
test_set['BsmtUnfSF']=test_set['BsmtUnfSF'].fillna(test_set['BsmtUnfSF'].mean())
test_set['TotalBsmtSF']=test_set['TotalBsmtSF'].fillna(test_set['TotalBsmtSF'].mean())
test_set['BsmtFullBath']=test_set['BsmtFullBath'].fillna(test_set['BsmtFullBath'].mode()[0])
test_set['BsmtHalfBath']=test_set['BsmtHalfBath'].fillna(test_set['BsmtHalfBath'].mode()[0])
test_set['KitchenQual']=test_set['KitchenQual'].fillna(test_set['KitchenQual'].mode()[0])
test_set['Functional']=test_set['Functional'].fillna(test_set['Functional'].mode()[0])
test_set['GarageCars']=test_set['GarageCars'].fillna(test_set['GarageCars'].mean())
test_set['GarageArea']=test_set['GarageArea'].fillna(test_set['GarageArea'].mean())
test_set['SaleType']=test_set['SaleType'].fillna(test_set['SaleType'].mode()[0])

test_set.to_csv('formulatedtest.csv',index=False)
