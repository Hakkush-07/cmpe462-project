from math import exp
import pandas as pd
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.001, iterations=500):
        self.learning_rate = learning_rate
        self.iterations = iterations

    @staticmethod
    def sigmoid(s):
        return 1 / (1 + exp(-s))
    
    def fit(self, x, y):
        self.m, self.n = x.shape
        print(self.m, self.n)
        self.w = np.zeros(self.n)
        self.b = 0
        self.x = x
        self.y = y

        for _ in range(self.iterations):
            A = 1 / (1 + np.exp(-(self.x.dot(self.w) + self.b)))
            print("A")
            print(A)
            print("y")
            print(self.y)
            print("y.T")
            print(self.y.T)  
            tmp = (A - self.y.T)         
            tmp = np.reshape(tmp, self.m)         
            dW = np.dot(self.x.T, tmp) / self.m          
            db = np.sum(tmp) / self.m  
            self.w -= self.learning_rate * dW     
            self.b -= self.learning_rate * db 

    def predict(self, x):
        z = 1 / (1 + np.exp(-(x.dot(self.w) + self.b)))         
        y = np.where(z > 0.5, 1, -1)         
        return y

    @staticmethod
    def compare(prediction, actual):
        total = actual.size
        correct = 0
        for i in range(total):
            if actual[i] == prediction[i]:
                correct += 1
        return correct / total

def logistic_regression():
    df = pd.read_excel("nba.xlsx")
    del df["Player Name"]
    df = df[[c for c in df.columns if c != "Position"] + ["Position"]]
    # df["Position"] = df["Position"].apply(lambda x: "Center" if x.startswith("Center") else "Guard" if x.startswith("Guard") else "Forward")
    # df["Position"] = df["Position"].apply(lambda x: [][["Guard", "Forward", "Center"].index(x)])
    df["Position"] = df["Position"].apply(lambda x: [1, 1, 1, -1, -1, -1, -1][["Guard", "Guard-Forward", "Forward-Guard", "Forward", "Forward-Center", "Center-Forward", "Center"].index(x)])

    test = df.sample(int(0.2 * df.shape[0]))
    train = df.drop(test.index, axis=0)
    test_x = test.iloc[:,:-1].values
    test_y = test.iloc[:,-1:].values
    train_x = train.iloc[:,:-1].values
    train_y = train.iloc[:,-1:].values
    print(train_x)
    print(train_y)

    model = LogisticRegression()
    model.fit(train_x, train_y)
    prediction_y = model.predict(test_x)
    result = LogisticRegression.compare(prediction_y, test_y)
    print(result)

    

logistic_regression()


