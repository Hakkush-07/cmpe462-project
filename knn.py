from math import exp, sqrt
import pandas as pd
import numpy as np

def cossim(a, b):
    return a.dot(b) / (sqrt(a.dot(a)) * sqrt(b.dot(b)))

class KNN:
    def __init__(self):
        pass

    def find_k(self, fold_count=5):
        possible_k = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        

    def fit(self, x, y, k=None):
        self.x = x
        self.y = y
        self.k = self.find_k() if k is None else k

    def predict(self, x):
        prediction = []
        for a in x:
            similarities = sorted([(cossim(a, b), c) for b, c in zip(self.x, self.y)])[::-1][:self.k]
            s = list(map(lambda x: x[1][0], similarities))
            m = max(set(s), key=s.count)
            prediction.append(m)
        return prediction

    @staticmethod
    def compare(prediction, actual):
        total = actual.size
        correct = 0
        for i in range(total):
            if actual[i] == prediction[i]:
                correct += 1
        return correct / total

def knn():
    df = pd.read_excel("nba.xlsx")
    del df["Player Name"]
    df = df[[c for c in df.columns if c != "Position"] + ["Position"]]
    df["Position"] = df["Position"].apply(lambda x: "Center" if x.startswith("Center") else "Guard" if x.startswith("Guard") else "Forward")

    # s = set()
    # for a in df["Position"]:
    #     s.add(a)
    # print(s)
    
    test = df.sample(int(0.2 * df.shape[0]))
    train = df.drop(test.index, axis=0)
    test_x = test.iloc[:,:-1].values
    test_y = test.iloc[:,-1:].values
    train_x = train.iloc[:,:-1].values
    train_y = train.iloc[:,-1:].values
    # print(train_x)
    # print(train_y)

    model = KNN()
    model.fit(train_x, train_y, 100)
    prediction_y = model.predict(test_x)
    print(prediction_y, test_y)
    result = KNN.compare(prediction_y, test_y)
    print(result)


knn()

