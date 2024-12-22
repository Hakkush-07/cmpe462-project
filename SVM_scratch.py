
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC,SVC
import numpy as np
from sklearn.preprocessing import StandardScaler
from cvxopt import matrix, solvers
from util import read_and_process
import time


class SVM_scratch:
    def __init__(self, C=1.0, tol=1e-4):
        self.C = C
        self.tol = tol
        self.weights = None
        self.bias = None

    def fit(self, X, y, C=1.0, tol=1e-4):
        n, features = X.shape
        P = np.multiply(np.outer(y, y), np.dot(X, X.T))
        q = np.full((n, 1), -1)
        G = np.vstack((-np.eye(n), np.eye(n)))
        h = np.hstack((np.zeros(n), np.full(n, C))).reshape(-1, 1)
        A = y.reshape(1, n)
        b = 0.0

        solution = solvers.qp(matrix(P.astype(float)), matrix(q.astype(float)), matrix(G.astype(float)),
                              matrix(h.astype(float)), matrix(A.astype(float)), matrix(b))
        alpha_values = np.array(solution['x']).ravel()
        w = np.sum((alpha_values * y)[:, None] * X, axis=0)
        sv_indices = (alpha_values > tol).ravel()
        sv = X[sv_indices]
        sv_labels = y[sv_indices]
        bias = np.mean(sv_labels - np.dot(sv, w))

        self.weights = w
        self.bias = bias

    def predict(self, x):
        return np.matmul(x, self.weights) + self.bias


data = read_and_process()

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values




def train_and_predict(X,y):
    st_x = StandardScaler()
    X = st_x.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, shuffle=True, test_size=0.2)
    y_train_guard = np.where(y_train == 0, 1, -1)
    y_train_forward = np.where(y_train == 1, 1, -1)
    y_train_center = np.where(y_train == 2, 1, -1)
    model_guard = SVM_scratch()
    model_forward = SVM_scratch()
    model_center = SVM_scratch()

    start_time = time.time()
    model_guard.fit(X_train, y_train_guard)
    model_forward.fit(X_train, y_train_forward)
    model_center.fit(X_train, y_train_center)
    end_time = time.time()

    diff_qp_svm_linear = end_time - start_time

    prediction_guard = model_guard.predict(X_test)
    prediction_forward = model_forward.predict(X_test)
    prediction_center = model_center.predict(X_test)

    predictions = []
    for g, f, c in zip(prediction_guard, prediction_forward, prediction_center):
        predictions.append(int(np.argmax([g, f, c])))
    predictions = np.array(predictions)

    accuracy = accuracy_score(y_test, predictions)

    # Scikit implementations


    clf = LinearSVC()
    start_time = time.time()
    clf.fit(X_train,y_train)
    end_time = time.time()
    diff_sk_svm_linear = end_time - start_time
    predictions = clf.predict(X_test)
    accuracy_linear_svc_sklearn = accuracy_score(y_test, predictions)

    svm = SVC()
    param_grid = {
        'C' : [0.001,0.01,0.1,1,10],
        'tol': [1e-2,1e-3,1e-4,1e-5],
        'kernel': ['linear','poly','rbf','sigmoid'],
        'gamma': [1e-3,1e-2,1e-1,1]
    }
    grid=GridSearchCV(svm,param_grid=param_grid,cv=5)
    grid.fit(X_train,y_train)

    print("Best Parameters:", grid.best_params_)
    print("Best Score:", grid.best_score_)


    return accuracy, predictions

train_and_predict(X,y)












