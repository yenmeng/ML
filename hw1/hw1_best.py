import sys
import os
import numpy as np
import pandas as pd
import csv
import math

testfilepath = sys.argv[1]
outputpath = sys.argv[2]

#read testfile
testdata = pd.read_csv(testfilepath, header = None, encoding = 'big5')

drop = ['RAINFALL','PM10','WS_HR']
for item in drop:
    testdata.drop(testdata[testdata[1]==item].index,inplace=True)

std_x = np.load('std_x_best.npy')
mean_x = np.load('mean_x_best.npy')

test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 15*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[15 * i: 15* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

w = np.load('weight_best.npy')
ans_y = np.dot(test_x, w)
ans_y[ans_y<0] = 0

"""# **Save Prediction to CSV File**"""
os.makedirs(os.path.dirname(outputpath), exist_ok=True)
with open(outputpath, mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)
