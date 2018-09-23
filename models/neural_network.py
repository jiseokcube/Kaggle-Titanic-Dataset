#Change to Stratified
#Fill age with averages, embarked with S

import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data into numpy array (numeric data only)
sex = lambda x: 0.0 if x == b"male" else 1.0  # male = 0, female = 1
embarked = lambda x: 0.0 if x == b'C' else 2.0 if x == b'Q' else 1.0 # C = 0, S = 1, Q = 2; fill missing embarked values with S
data = np.genfromtxt(open("../data/train.csv"), delimiter=',', skip_header=1, usecols=(1, 2, 5, 6, 7, 8, 10, 12), converters={5: sex, 12: embarked})

# Fill missing age and fare (testing data only) values with averages for each class
def fill_missing(data):
    class1 = data[:, 0] == 1
    class2 = data[:, 0] == 2
    class3 = data[:, 0] == 3

    avg_age1 = np.nanmean(data[class1, 2])
    avg_age2 = np.nanmean(data[class2, 2])
    avg_age3 = np.nanmean(data[class3, 2])
    avg_fare1 = np.nanmean(data[class1, 5])
    avg_fare2 = np.nanmean(data[class2, 5])
    avg_fare3 = np.nanmean(data[class3, 5])

    age_nans = np.isnan(data[:, 2])
    fare_nans = np.isnan(data[:, 5])

    data[age_nans & class1, 2] = avg_age1
    data[age_nans & class2, 2] = avg_age2
    data[age_nans & class3, 2] = avg_age3
    data[fare_nans & class1, 5] = avg_fare1
    data[fare_nans & class2, 5] = avg_fare2
    data[fare_nans & class3, 5] = avg_fare3

# Neural network model
X = data[:, 1:]
y = data[:, 0]
fill_missing(X)

scaler = StandardScaler()
nn = MLPClassifier(hidden_layer_sizes=(20, 20, 20, 20, 20), solver="sgd", max_iter=1500)
pipeline = Pipeline([("scaler", scaler), ("nn", nn)])
tuned_params = {"nn__alpha": 10.0 ** -np.arange(1, 7), "nn__batch_size": 2 ** np.arange(4, 9), "nn__learning_rate_init": 10.0 ** -np.arange(1, 7)}
kf = StratifiedKFold(n_splits=5)

clf = GridSearchCV(pipeline, tuned_params, n_jobs=7, cv=kf)
clf.fit(X, y)

print(clf.best_params_)
means = clf.cv_results_["mean_test_score"]
stds = clf.cv_results_["std_test_score"]
for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))



"""
train_accs = []
test_accs = []

for train_index, test_index in kf.split(data): # loop over each fold
    train_data = data[train_index]
    test_data = data[test_index]
    train_X = train_data[:, 1:]
    train_y = train_data[:, 0]
    test_X = test_data[:, 1:]
    test_y = test_data[:, 0]

    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    clf.fit(train_X, train_y)

    train_pred = clf.predict(train_X)
    train_comp = train_pred == train_y
    train_acc = sum(train_comp) / len(train_comp)
    train_accs.append(train_acc)

    test_pred = clf.predict(test_X)
    test_comp = test_pred == test_y
    test_acc = sum(test_comp) / len(test_comp)
    test_accs.append(test_acc)

print(train_accs)
print(test_accs)
"""
