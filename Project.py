import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
train_df = pd.read_csv('C:\\Users\\my\\Desktop\\Git projects\\Project 1 - Mercedes-Benz Greener Manufacturing\\Dataset for the project\\train.csv')
test_df = pd.read_csv('C:\\Users\\\my\\Desktop\\Git projects\\Project 1 - Mercedes-Benz Greener Manufacturing\\Dataset for the project\\test.csv')
print(train_df.shape)

print(train_df.columns)

print(test_df.shape)

print(test_df.columns)

train_df.head()

test_df.head()

train_df.describe()

test_df.describe()

train_df.var()

train_df.var()== 0

(train_df.var()== 0).values

var_zero = train_df.var()[train_df.var()==0].index.values

var_zero.shape

train_df= train_df.drop(var_zero,axis = 1)

train_df.shape

train_df = train_df.drop(['ID'],axis = 1)

train_df.head()

train_df.isnull().sum().values

train_df.isnull().any()

train_df.nunique()

obj_dtype = train_df.select_dtypes(include=[object])
obj_dtype

label_encoder = preprocessing.LabelEncoder()
train_df['X0'].unique()

train_df['X0'] = label_encoder.fit_transform(train_df['X0'])
train_df['X0'].unique()

train_df['X1'] = label_encoder.fit_transform(train_df['X1'])
train_df['X2'] = label_encoder.fit_transform(train_df['X2'])
train_df['X3'] = label_encoder.fit_transform(train_df['X3'])
train_df['X4'] = label_encoder.fit_transform(train_df['X4'])
train_df['X5'] = label_encoder.fit_transform(train_df['X5'])
train_df['X6'] = label_encoder.fit_transform(train_df['X6'])
train_df['X8'] = label_encoder.fit_transform(train_df['X8'])


train_df.head()

from sklearn.decomposition. import PCA

skl_pca = PCA(n_components= 0.95)
skl_pca.fit(train_df)

x_train_trans = skl_pca.transform(train_df)
x_train_trans.shape

pca_98 = PCA(n_components = 0.98)
pca_98.fit(train_df)

x_pca_98 = pca_98.transform(train_df)
print(x_pca_98.shape)

train_df.y

x= train_df.drop('y',axis = 1)
y = train_df.y

xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3,random_state = 42)
print(xtrain)
print(xtrain.shape)


print(ytrain)
print(ytest.shape)

print(xtest)
print(xtest.shape)

pca_xtrain= PCA(n_components =0.95)

pca_xtrain.fit(xtrain)

pca_xt_trans=pca_xtrain.transform(xtrain)
print(pca_xt_trans.shape)

pca_xtest= PCA(n_components =0.95)
pca_xtest.fit(xtest)

pca_xtest_trans= pca_xtest.transform(xtest)
print(pca_xtest_trans.shape)

print(pca_xtest.explained_variance_)
print(pca_xtest.explained_variance_ratio_)

test_df

test_obj_dtype = test_df.select_dtypes(include = [object])
test_obj_dtype

test_df['X0']=label_encoder.fit_transform(test_df['X0'])
test_df['X1']=label_encoder.fit_transform(test_df['X1'])
test_df['X2']=label_encoder.fit_transform(test_df['X2'])
test_df['X3']=label_encoder.fit_transform(test_df['X3'])
test_df['X4']=label_encoder.fit_transform(test_df['X4'])
test_df['X5']=label_encoder.fit_transform(test_df['X5'])
test_df['X6']=label_encoder.fit_transform(test_df['X6'])
test_df['X8']=label_encoder.fit_transform(test_df['X8'])

print(test_df)
print(test_df.shape)

test_df =test_df.drop('ID',axis = 1)
pca_test_df = PCA(n_components = 0.95)
pca_test_df.fit(test_df)

pca_tst_df_trans = pca_test_df.transform(test_df)
print(pca_tst_df_trans.shape)

print(pca_test_df.explained_variance_)
print(pca_test_df.explained_variance_ratio_)

from sklearn import svm
from sklearn import model_selection
import xgboost as xgb

model = xgb.XGBRegressor(objective="reg:linear",learning_rate=0.20)
model.fit(pca_xtrain,ytrain)
y_pred=model.predict(pca_xtest)
y_pred
model.predict(pca_test_df)