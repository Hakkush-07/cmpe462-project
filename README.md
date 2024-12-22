# CMPE462 Machine Learning Project

Team: Ludos (Hakan Karakuş, Melih Özcan)

In this project, we train machine learning models to predicts the position (Guard, Forward, Center) of basketball players based on the match statistics.

## Data Collection

to install dependencies (you may need to get a webdriver as well)

`pip install selenium openpyxl lxml pandas beautifulsoup4`

The final data used in the models is in `nba.xlsx`. To produce it, first use `main.py` to scrape stats. Also use `scrape_nba_players.py` to get player specific information. Then run `process.py` and `process2.py` to get the final form of the data.

## Models

to install dependencies

`pip install scikit-learn seaborn matplotlib`

### Logistic Regression

`logistic_regression.py` contains both the scikit-learn and our from scratch implementations. A correlation table and confusion matrices can be produces by the respective functions in here.

### SVM

`SVM_scratch.py` contains both the scikit-learn and our from scratch implementations. 

### kNN

`kNN.py` contains both the scikit-learn and our from scratch implementations. 

### Random Forest

`RandomForest.py` contains the applications of the Random Forest Classifier algorithm and the hyperparameter tuning for it along with various metrics.

### Plotting

`plots.py` produces the plots for the decision boundries when a binary Logistic Regression and SVM model are trained for two chosen attributes and classes. 


