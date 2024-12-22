import math

import pandas as pd
import numpy as np

import time

from sklearn.metrics import accuracy_score, average_precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import util


# Do 5-fold cross validation, take the mean of prediction accuracy rate and decide the best k-value according
# to the accuracy score
def cross_validate(X,y,k_values,fold = 5):
    kf = KFold(n_splits=fold,shuffle=True,random_state=42)
    scores = {k: [] for k in k_values}
    for train_indices,test_indices in kf.split(X):
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        for k in k_values:
            correct_predictions = 0
            test_size = len(X_test)
            for i in range(test_size):
                prediction = kNN(X_train,y_train,X_test[i],k)
                if(prediction == y_test[i]):
                    correct_predictions += 1
            acc = correct_predictions / test_size
            scores[k].append(acc)
    avg_scores = {k: np.mean(scores[k]) for k in k_values}
    best_k = max(avg_scores, key= avg_scores.get)
    return best_k

# Function that calculates the eucledian distance between two numpy arrays (vectors)
def calculate_eucledian_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# kNN algorithm implementation from scratch, for the point that needs its category be determined
# calculate the eucledian distance between that point and all the points from the training set
# sort the distances and take the k points with the least distance
# determine the class of the point according to the majority vote of which class has the most instances
# inside the sorted array
def kNN(X_train, y_train, single_point_x, k ):
    size = len(X_train)
    result_array = []
    for i in range(size):
        compared_data = X_train[i]
        compared_category = y_train[i]
        dist = calculate_eucledian_distance(single_point_x,compared_data)
        result_array.append((dist,compared_category))
    sorted_diff = sorted(result_array,key=lambda x: x[0])
    sorted_sliced = sorted_diff[:k]
    category_occurrences = []
    for tp in sorted_sliced:
        category_occurrences.append(tp[1])
    return max(set(category_occurrences),key=category_occurrences.count)

data = util.read_and_process()

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# Read and scale the data
st_x = StandardScaler()
X = st_x.fit_transform(X)

k_values = [1,3,5,7,9]
best_k = cross_validate(X,y,k_values)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, shuffle=True, test_size=0.2)

# Best k is 9
start_time = time.time()
predictions = [kNN(X_train,y_train,X_test[i],best_k) for i in range(len(y_test))]
end_time = time.time()
duration_self_knn = end_time - start_time
print('Duration knn from scratch: ', duration_self_knn)
accuracy = accuracy_score(y_test,predictions)
recall = recall_score(y_test, predictions,average='macro')
f1_score= f1_score(y_test, predictions,average='macro')






