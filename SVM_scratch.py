from sklearn.metrics import accuracy_score, average_precision_score, recall_score, roc_auc_score, f1_score
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

# The method that trains a model and calculates the optimal weights and bias term for linear soft margin SVM
    def fit(self, X, y, C=1.0, tol=1e-4):
        # We want to find the z term which is the alpha values (Lagrange multipliers) in our case
        n, features = X.shape
        # P corresponds to the yiyj(XiTXj) term. This is essentially the element-wise multiplication of the outer product of y vector
        # with the multiplication of X transpose with X
        P = np.multiply(np.outer(y, y), np.dot(X, X.T))
        # Since we convert the maximization problem in the dual formulation into a minimization problem,
        # we multiply it with -1 and the signs change. qTz should equal -(sum of all alpha vales) and as a result q corresponds to a vector of size n with all -1s as values
        q = np.full((n, 1), -1)
        # 0 <= ai <= C is the inequality constraint in our case and can be broken down into -ai <= 0 and
        # ai <= C, in order to get this in a matrix multiplication form like Gz <= h, we set the above half of
        # matrix G to an identity matrix of size n multiplied by -1 and the below half an identity matrix of size
        # n. (2nxn matrix) This is to ensure that we capture the above two inequalities when we multiply it with
        # alpha values. (z) And again we set the first n terms of h to 0 (-ai <= 0) and the rest n terms to C
        # C. (ai <= C)
        G = np.vstack((-np.eye(n), np.eye(n)))
        h = np.hstack((np.zeros(n), np.full(n, C))).reshape(-1, 1)
        # The last constraint is Az = b. We have the sum of all alphai * yi = 0. So the A is the
        # transpose of y and b = 0.
        A = y.reshape(1, n)
        b = 0.0


        # Feed the terms to the qp solver get the alpha values, calculate the weights and then the biases
        # using the support vectors.
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
    prediction_probas = []
    for g, f, c in zip(prediction_guard, prediction_forward, prediction_center):
        predictions.append(int(np.argmax([g, f, c])))
        prediction_probas.append([g,f,c])
    predictions = np.array(predictions)
    prediction_probas= np.array(prediction_probas)

    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions, average='macro')
    f1_scored = f1_score(y_test, predictions, average='macro')

    print('Accuracy, recall, f1-score and runtime for svm from scratch: ', acc,recall,f1_scored,diff_qp_svm_linear)
    # Scikit implementations

    clf = LinearSVC()
    start_time = time.time()
    clf.fit(X_train,y_train)
    end_time = time.time()
    diff_sk_svm_linear = end_time - start_time
    predictions = clf.predict(X_test)
    train_predictions = clf.predict(X_train)
    accuracy_linear_svc_sklearn = accuracy_score(y_test, predictions)
    accuracy_linear_svc_sklearn_train = accuracy_score(y_train, train_predictions)
    recall = recall_score(y_test, predictions, average='macro')
    f1_scored = f1_score(y_test, predictions, average='macro')

    print('Train accuracy, test accuracy, recall, f1-score and runtime for linear SVM: ', accuracy_linear_svc_sklearn_train, accuracy_linear_svc_sklearn, recall, f1_scored, diff_qp_svm_linear)
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


    clf_optimal = SVC(C=10, kernel='rbf', tol=0.01, gamma=0.01)
    start_time = time.time()
    clf_optimal.fit(X_train,y_train)
    end_time = time.time()
    diff = end_time - start_time
    predictions = clf_optimal.predict(X_test)
    predictions_train = clf_optimal.predict(X_train)
    acc_train = accuracy_score(y_train, predictions_train)
    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions, average='macro')
    f1_scored = f1_score(y_test, predictions, average='macro')

    print('Train accuracy, test accuracy, recall, f1-score and runtime for linear SVM: ',
          acc_train, acc, recall, f1_scored, diff)

train_and_predict(X,y)
