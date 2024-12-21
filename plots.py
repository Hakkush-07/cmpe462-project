import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_excel("nba.xlsx")

position_dct = {
    "Guard": 1,
    "Guard-Forward": 1,
    "Forward-Guard": 2,
    "Forward": 2,
    "Forward-Center": 2,
    "Center-Forward": 3,
    "Center": 3,
}

del df["Player Name"]
df = df[[c for c in df.columns if c != "Position"] + ["Position"]]
print(df["Position"].unique())
print(df.isnull().sum())
df["Blocks Contribution"] = df["Blocks Contribution"].fillna(0)
df["Position"] = df["Position"].apply(lambda x: position_dct[x])

a, b = "3 Pointers Attempted Per Match", "Rebounds Per Match"

# low, high = 0.05, 0.95
# for s in [a, b]:
#     sx = list(df[s].quantile([low, high]))
#     df = df.loc[(df[s] > sx[0]) & (df[s] < sx[1])]

centers = df[df["Position"] == 3]
guards = df[df["Position"] == 1]

size = 10
plt.scatter(centers[a], centers[b], c="blue", s=size)
plt.scatter(guards[a], guards[b], c="red", s=size)

plt.xlabel(a)
plt.ylabel(b)
ca = plt.gca()

class LR:
    def __init__(self, learning_rate=0.001, iterations=2000):
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

def get_weights_and_bias_for_binary_logistic_regression():
    df = pd.read_excel("nba.xlsx")
    del df["Player Name"]
    df = df[[c for c in df.columns if c != "Position"] + ["Position"]]
    df["Blocks Contribution"] = df["Blocks Contribution"].fillna(0)
    df["Position"] = df["Position"].apply(lambda x: position_dct[x])

    df = df[(df["Position"] == 3) | (df["Position"] == 1)]
    df = df[[a, b, "Position"]]

    X = df.drop("Position", axis=1)
    y = df["Position"]
    st_x = StandardScaler()
    X = st_x.fit_transform(X)

    y = y.apply(lambda x: 1 if x == 1 else -1)
    model = LR()
    model.fit(X, y)
    return model.weights, model.bias

weights, bias = get_weights_and_bias_for_binary_logistic_regression()
print(weights, bias)

decision_boundry_x = np.linspace(0, max(df[a]))
decision_boundry_y = (-weights[0] * decision_boundry_x - bias) / weights[1]
plt.plot(decision_boundry_x, decision_boundry_y, 'k-')

plt.show()

