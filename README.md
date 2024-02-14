# UW-Madison-STAT451
## UW Madison STAT451 Introduction to Machine Learning and Statistical Pattern Classification in Summer 2023
Instructor: John Gillett

*Across all assignments, `matplotlib`, `numpy`, and `pandas` were used to handle graphing, inputs, and importation of data respectively.* 

## Homework 1:

Utilized Scikit-Learn's library for SVM and Regression Models:

- `svm.SVC`: to create a hard-margin SVM model to classify cars as having either automatic or manual transmissions
- `linear_model.LinearRegression`:
    - a simple linear regression model predicting the average daily trading volume of a Dow Jones Industrial Average stock using its market capitalization
    - a multiple regression model to predict the average daily trading volume again but with market capitalization and price

## Homework 2: 

Utilized Scikit-Learn's library for Logistic Regression and Decision Trees:

- 'linear_model.LogisticRegression()': Using the Iris dataset, predicted whether an iris was virginica or not using petal length.
-  `DecisionTreeClassifier()`: Used a Titanic dataset to predict whether a passenger survived or not using sex, age, and ticket class

## Homework 3:

Utilized Scikit-Learn's library for SVM, KNN, Gradient Descent and performed feature engineering:

- Created a decision tree, knn, and a linear and nonlinear SVM model; visualized them using `matplotlib` to compare their classification decisions
- Rescaled data using Scikit-Learn's `StandardScaler()` to improve KNN accuracy.

## Homework 4:

Practiced feature engineering:

- Applied one hot encoding and data imputation to data using `pd.get_dummies` and Scikit-Learn's `SimpleImputer()`
- Compared models from Scikit-Learn's `LinearRegression()`, `Lasso()`, and `Ridge` to evaluate fit, overfitting, and regularization in multiple linear regression.

## Homework 5:

Practiced algorithm selection, assessment, hypterparameter tuning, multiclass and one-class classification, and handled imbalanced data

- Selected an algorithm for multiclass classification for the recognition of handwritten digits using the `digits` dataset; examined SVM, logistic regression, decision tree, and knn
- Performed outlier detection using a gradebook dataset and `mixture.GaussianMixture()` to create a Gaussian model for this purpose
- Explored how accuracy can be misleading using imbalanced data
- Used `RandomOverSampler()` to balance dataset and trained a `GradientBoostingClassifier()` on it and reported True and False positive rate

