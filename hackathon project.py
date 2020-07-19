# Hackathon Project

#PART I- Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#IMPORTING THE DATASET
train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

# CREATING MATRIX OF DATASET
X_train = train_set.iloc[:, 1:80].values
Y_train = train_set.iloc[:, 80].values

# FILL MISSING VALUES
from sklearn.impute import SimpleImputer
train_set['LotFrontage'] = train_set['LotFrontage'].fillna(train_set['LotFrontage'].mean())
train_set['BsmtQual'] = train_set['BsmtQual'].fillna(train_set['BsmtQual'].mode()[0])
train_set['BsmtCond'] = train_set['BsmtCond'].fillna(train_set['BsmtCond'].mode()[0])
train_set['FireplaceQu'] = train_set['FireplaceQu'].fillna(train_set['FireplaceQu'].mode()[0])
train_set['GarageType'] = train_set['GarageType'].fillna(train_set['GarageType'].mode()[0])
train_set['GarageFinish'] = train_set['GarageFinish'].fillna(train_set['GarageFinish'].mode()[0])
train_set['GarageQual'] = train_set['GarageQual'].fillna(train_set['GarageQual'].mode()[0])
train_set['GarageCond'] = train_set['GarageCond'].fillna(train_set['GarageCond'].mode()[0])
train_set['MasVnrType'] = train_set['MasVnrType'].fillna(train_set['MasVnrType'].mode()[0])
train_set['MasVnrArea'] = train_set['MasVnrArea'].fillna(train_set['MasVnrArea'].mode()[0])
train_set['BsmtExposure'] = train_set['BsmtExposure'].fillna(train_set['BsmtExposure'].mode()[0])
train_set['BsmtFinType2'] = train_set['BsmtFinType2'].fillna(train_set['BsmtFinType2'].mode()[0])
train_set.drop(['Alley'], axis = 1, inplace = True)
train_set.drop(['GarageYrBlt'], axis = 1, inplace = True)
train_set.drop(['PoolQC', 'Fence', 'MiscFeature'], axis = 1, inplace = True)
train_set.drop(['Id'], axis = 1, inplace = True)
train_set.dropna(inplace = True)

# ENCODING CATEGORICAL DATA


