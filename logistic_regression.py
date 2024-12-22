import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

def read_and_process():
    df = pd.read_excel("nba.xlsx")
    del df["Player Name"]
    df = df[[c for c in df.columns if c != "Position"] + ["Position"]]
    df["Position"] = df["Position"].apply(lambda x: "Center" if x.startswith("Center") else "Guard" if x.startswith("Guard") else "Forward")
    print(df["Position"].unique())
    print(df.isnull().sum())
    df["Blocks Contribution"] = df["Blocks Contribution"].fillna(0)
    df["Position"] = df["Position"].apply(lambda x: ["Guard", "Forward", "Center"].index(x))
    # for s in ["Points", "Rebounds", "Offensive Rebounds", "Defensive Rebounds", "Assists", "Steals", "Blocks", "Turnovers", "Field Goals Attempted", "Field Goals Made", "Free Throws Attempted", "Free Throws Made", "3 Pointers Attempted", "3 Pointers Made"]:
    #     df = df.drop([f"{s} Per Match"], axis=1)
    return df

def plot_classes(df):
    sns.countplot(x="Position", data=df)
    plt.xticks(rotation=45)
    plt.show()

def plot_correlation(df):
    plt.figure(figsize=(15, 10))
    sns.heatmap(df.corr(), annot=True)
    plt.show()

def plot_confusion(y_test, predictions):
    print(confusion_matrix(y_test, predictions))
    cm = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Guard", "Forward", "Center"])
    disp.plot()
    plt.show()

def get_test_train(df):
    X = df.drop("Position", axis=1)
    y = df["Position"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, shuffle=True, test_size=0.2)
    st_x = StandardScaler()
    X_train = st_x.fit_transform(X_train)
    X_test = st_x.transform(X_test)
    return X_train, X_test, y_train, y_test

def scikit_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(random_state=42, solver="newton-cg", multi_class="multinomial", n_jobs=1, C=1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, predictions

class LR:
    def __init__(self, learning_rate=0.001, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    @staticmethod
    def sigmoid(s):
        return 1 / (1 + np.exp(-s))

    def fit(self, x, y):
        count, features = x.shape
        self.weights = np.zeros(features)
        self.bias = 0
        
        for _ in range(self.iterations):
            pr = LR.sigmoid(x.dot(self.weights) + self.bias)

            dw = (1 / count) * x.T.dot(pr - y)
            db = (1 / count) * np.sum(pr - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, x):
        return LR.sigmoid(x.dot(self.weights.T) + self.bias)

def my_logistic_regression(X_train, X_test, y_train, y_test):
    y_train_guard   = y_train.apply(lambda x: 1 if x == 0 else -1)
    y_train_forward = y_train.apply(lambda x: 1 if x == 1 else -1)
    y_train_center  = y_train.apply(lambda x: 1 if x == 2 else -1)

    model_guard   = LR()
    model_forward = LR()
    model_center  = LR()

    model_guard.fit(X_train, y_train_guard)
    model_forward.fit(X_train, y_train_forward)
    model_center.fit(X_train, y_train_center)

    prediction_guard   = model_guard.predict(X_test)
    prediction_forward = model_forward.predict(X_test)
    prediction_center  = model_center.predict(X_test)

    predictions = []
    for g, f, c in zip(prediction_guard, prediction_forward, prediction_center):
        predictions.append(int(np.argmax([g, f, c])))
    predictions = np.array(predictions)
    
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy, predictions

def main():
    df = read_and_process()
    # plot_classes(df)
    # plot_correlation(df)
    X_train, X_test, y_train, y_test = get_test_train(df)
    accuracy_scikit, predictions_scikit = scikit_logistic_regression(X_train, X_test, y_train, y_test)
    accuracy_my, predictions_my = my_logistic_regression(X_train, X_test, y_train, y_test)
    # plot_confusion(y_test, predictions_scikit)
    plot_confusion(y_test, predictions_my)
    print(f"scikit: {accuracy_scikit}")
    print(f"my    : {accuracy_my}")

if __name__ == "__main__":
    main()

