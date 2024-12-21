from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import time

import util

data = util.read_and_process()

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
st_x = StandardScaler()
X = st_x.fit_transform(X)

clf = RandomForestClassifier()
grid_search = GridSearchCV(clf,param_grid={'n_estimators': [20,50,100,200,400]}, cv=5)

grid_search.fit(X,y)

best_param = grid_search.best_params_['n_estimators']
clf1 = RandomForestClassifier(n_estimators=best_param)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, shuffle=True, test_size=0.2)
start_time = time.time()
clf1.fit(X_train,y_train)
end_time= time.time()
diff_time = end_time - start_time

predictions = clf1.predict(X_test)
probabilities = clf1.predict_proba(X_test)
acc = accuracy_score(y_test, predictions)
avg = average_precision_score(y_test, probabilities,average='macro')
recall = recall_score(y_test, predictions,average='macro')
f1_score= f1_score(y_test, predictions,average='macro')
roc_auc = roc_auc_score(y_test,probabilities,average='macro',multi_class='ovr')
print()
