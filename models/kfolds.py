import numpy as np
from sklearn.model_selection import KFold

# Load data into numpy array (numeric data only)
sex = lambda x: 0.0 if x == b'male' else 1.0  # male = 0, female = 1
embarked = lambda x: 0.0 if x == b'C' else 1.0 if x == b'S' else 2.0 # C = 0, S = 1, Q = 2
data = np.genfromtxt(open("../data/train.csv"), delimiter=',', skip_header=1, usecols=(1, 2, 5, 6, 7, 8, 10, 12), converters={5: sex, 12: embarked})

# Fill missing age values with averages for each class
class1 = data[:, 1] == 1
class2 = data[:, 1] == 2
class3 = data[:, 1] == 3
avg_age1 = np.nanmean(data[class1, 3])
avg_age2 = np.nanmean(data[class2, 3])
avg_age3 = np.nanmean(data[class3, 3])
nans = np.isnan(data[:, 3])
data[nans & class1, 3] = avg_age1
data[nans & class2, 3] = avg_age2
data[nans & class3, 3] = avg_age3

# Perform kfold of 5 splits
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(data): # loop over each fold
    train_data = data[train_index]
    test_data = data[test_index]
