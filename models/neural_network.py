import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix

# Load data into numpy array (numeric data only)
sex = lambda x: 0.0 if x == b"male" else 1.0  # male = 0, female = 1
embarked = lambda x: 0.0 if x == b'C' else 2.0 if x == b'Q' else 1.0 # C = 0, S = 1, Q = 2; fill missing embarked values with S
train_data = np.genfromtxt(open("../data/train.csv"), delimiter=',', skip_header=1, usecols=(1, 2, 5, 6, 7, 8, 10, 12), converters={5: sex, 12: embarked})
test_data = np.genfromtxt(open("../data/test.csv"), delimiter=',', skip_header=1, usecols=(0, 1, 4, 5, 6, 7, 9, 11), converters={4: sex, 11: embarked})

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

# Data
train_X = train_data[:, 1:]
train_y = train_data[:, 0]
test_ids = test_data[:, 0]
test_X = test_data[:, 1:]
fill_missing(train_X)
fill_missing(test_X)

# Neural network model
scaler = StandardScaler()
nn = MLPClassifier(hidden_layer_sizes=(20, 20, 20, 20, 20), solver="sgd", max_iter=2500, random_state=0)
pipeline = Pipeline([("scaler", scaler), ("nn", nn)])
tuned_params = {"nn__alpha": 10.0 ** -np.arange(1, 7), "nn__batch_size": 2 ** np.arange(4, 9), "nn__learning_rate_init": 10.0 ** -np.arange(1, 7)}
kf = StratifiedKFold(n_splits=5)

clf = GridSearchCV(pipeline, tuned_params, n_jobs=7, cv=kf)
clf.fit(train_X, train_y)

# Hyperparameter search results
print(clf.best_params_)
means = clf.cv_results_["mean_test_score"]
stds = clf.cv_results_["std_test_score"]
for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

# Training statistics
acc = clf.score(train_X, train_y)
print(acc)
train_pred = clf.predict(train_X)
conf_matrix = confusion_matrix(train_y, train_pred)
print(conf_matrix)
print()

# Output testing labels
test_pred = clf.predict(test_X)
output = np.stack((test_ids, test_pred), axis=1)
np.savetxt("submission.csv", output, fmt="%i", delimiter=',', header="PassengerId,Survived", comments='')
