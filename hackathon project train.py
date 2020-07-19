# Hackathon Project Train
# TEAM NAME - CODER GIRLS
# TEAM PATICIPANTS _
# Avneet Kaur
# Ruhee Jain
# Muskan Jaglan
# Sejal Jain
# Aanchal Rakheja

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#IMPORTING THE DATASET
train_set = pd.read_csv('train.csv')

train_set.shape

# FILL MISSING VALUES
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
columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope',
         'Neighborhood','Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond','ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure',
        'BsmtFinType1','BsmtFinType2','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
        'Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']
def category_onehot_multcols(multcolumns):
    train_set_final=final_train_set
    i=0
    for fields in multcolumns:
        
        print(fields)
        train_set_1=pd.get_dummies(final_train_set[fields],drop_first=True)
        
        final_train_set.drop([fields],axis=1,inplace=True)
        if i==0:
            train_set_final=train_set_1.copy()
        else:
            
            train_set_final=pd.concat([train_set_final,train_set_1],axis=1)
        i=i+1
       
        
    train_set_final=pd.concat([final_train_set,train_set_final],axis=1)
        
    return train_set_final

# Copying Training Set
main_train_set = train_set.copy()

# Combine Test Data
test_set = pd.read_csv('formulatedtest.csv')
final_train_set = pd.concat([train_set,test_set],axis=0,sort=True)

final_train_set.shape

# Applying Encoding Categorical Data to all the categorical columns
final_train_set = category_onehot_multcols(columns)

# Removing the duplicated columns
final_train_set =final_train_set.loc[:,~final_train_set.columns.duplicated()]

# Splitting the final set into train and test set
train_dataset = final_train_set.iloc[:1422,:]
test_dataset = final_train_set.iloc[1422:,:]
test_dataset.drop(['SalePrice'],axis=1,inplace=True)

# Dropping the dependent variable
X_train = train_dataset.drop(['SalePrice'],axis=1)
Y_train = train_dataset['SalePrice']

# PREDICTING THE RESULTS

# Creating classifier
import xgboost
classifier = xgboost.XGBRegressor()
classifier.fit(X_train, Y_train)

# Creating regressor
import xgboost
regressor = xgboost.XGBRegressor()

# Hyper Parameter Optimization
n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]
base_score=[0.25,0.5,0.75,1]

# Define the grid of hyperparameters to search
hyperparameter_grid = {'n_estimators': n_estimators,'max_depth':max_depth,'learning_rate':learning_rate,
                       'min_child_weight':min_child_weight,'booster':booster,'base_score':base_score}


# Set up the random search with 4-fold cross validation
from sklearn.model_selection import RandomizedSearchCV
random_cv = RandomizedSearchCV(estimator=regressor,param_distributions=hyperparameter_grid,cv=5,n_iter=50,
            scoring='neg_mean_absolute_error',n_jobs=4,verbose=5, return_train_score=True,random_state=0)

# Fitting the training set to random_cv
random_cv.fit(X_train,Y_train)

# Finding the best parameter values for optimizing the model
random_cv.best_estimator_

# Updating regressor with optimum parameters
regressor = xgboost.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=2, min_child_weight=1, missing=None, n_estimators=500,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)

# Fitting training set to our regressor
regressor.fit(X_train, Y_train)

# Predicting the test set results
Y_pred = regressor.predict(test_dataset)

# Create Sample Submission file and Submit using ANN
pred = pd.DataFrame(Y_pred)
sub_set = pd.read_csv('sample_submission.csv')
datasets = pd.concat([sub_set['Id'],pred],axis=1)
datasets.columns = ['Id','SalePrice']
datasets.to_csv('sample_submission.csv',index=False)

pred.columns = ['SalePrice']
temp_dataset = train_set['SalePrice'].copy()
temp_dataset.column = ['SalePrice']
train_set.drop(['SalePrice'],axis=1,inplace=True)
train_set = pd.concat([train_set,temp_dataset],axis=1)
test_set = pd.concat([test_set,pred],axis=1)
X_train = train_set.drop(['SalePrice'],axis=1)
Y_train = train_set['SalePrice']


# MAKE THE ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu', input_dim = 174))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 25, init = 'uniform', activation = 'relu'))

"""# Adding the third hidden layer
classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu'))"""

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(loss = 'binary_crossentropy', optimizer='adam')

# Fitting the ANN to the training set
classifier.fit(X_train, Y_train, validation_split=0.20, batch_size = 10, nb_epoch = 1000)