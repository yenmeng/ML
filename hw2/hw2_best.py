#!/usr/bin/env python
# coding: utf-8


import sys
import os
import numpy as np
np.random.seed(1) 
import pandas as pd
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model


X_train_fpath = sys.argv[3]
Y_train_fpath = sys.argv[4]
X_test_fpath = sys.argv[5]
output_fpath = sys.argv[6]

'''# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)'''
X_train = pd.read_csv(X_train_fpath)
X_train = X_train.drop('id',axis=1)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
X_test = pd.read_csv(X_test_fpath)
X_test = X_test.drop('id',axis=1)

numeric_cols = [col for col in X_train.columns if X_train[col].nunique() > 2 ]
#print(numeric_cols)
#numeric_cols.remove('wage per hour')
one_val_cols = [col for col in X_train.columns if X_train[col].nunique() <=1 ]
unknown_cols = [col for col in X_train.columns if '?' in col ]

for item in numeric_cols:
    name = item + str('_1')
    X_train[name] = X_train[item]*0.8
    X_test[name] = X_test[item]*0.8
for item in numeric_cols:
    name = item + str('_2')
    X_train[name] = X_train[item]*0.8
    X_test[name] = X_test[item]*0.8
for item in numeric_cols:
    name = item + str('_3')
    X_train[name] = X_train[item]*0.8
    X_test[name] = X_test[item]*0.8

X_train['wage_week_1'] = X_train['wage per hour']*X_train['weeks worked in year']
X_test['wage_week_1'] = X_test['wage per hour']*X_test['weeks worked in year']
X_train['wage_week_2'] = X_train['wage per hour']*X_train['weeks worked in year']
X_test['wage_week_2'] = X_test['wage per hour']*X_test['weeks worked in year']

# drop country
X_train = X_train.drop(X_train.iloc[:, 364:493], axis=1)
X_test = X_test.drop(X_test.iloc[:, 364:493], axis=1)

# drop cols
for i in unknown_cols:
    if i in X_train.columns:
        X_train[i] = X_train[i]*0.8
        X_test[i] = X_test[i]*0.8

X_train = X_train.to_numpy(dtype=float)
X_test = X_test.to_numpy(dtype=float)

def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
     
    return X, X_mean, X_std

def _train_dev_split(X, Y, dev_ratio = 0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train = True)
X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)
    
# Split data into training set and development set
dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio = dev_ratio)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
print('Size of training set: {}'.format(train_size))
print('Size of development set: {}'.format(dev_size))
print('Size of testing set: {}'.format(test_size))
print('Dimension of data: {}'.format(data_dim))


new_model = load_model('keras_model.h5')
re = new_model.predict(X_test)
re = (re>0.5).astype(int)

_id = np.arange(X_test.shape[0]).reshape(-1,1)
reproduce = np.concatenate((_id,re),axis=1)


sub = pd.DataFrame(data=reproduce,columns=['id','label'])

#best = pd.read_csv('../108-2/ml/hw2/best/ans.csv')
os.makedirs(os.path.dirname(output_fpath), exist_ok=True)
sub.to_csv(output_fpath,index=False)

'''
origin = list(best['label'].values)
reproduce = list(sub['label'].values)

print(origin == reproduce)
'''
