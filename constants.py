'''
In this file the hyperparameters for the Random Forest and Logistic Regression
model can be adjusted to have an easier interaction with the ChurnPredictor. The
parameters are imported to the churn_library.py in the beginning of the execution.

Author: David Hedderich
Date: 01.03.2023
'''

# Define your hyperparameter search space for the random forrest below
random_forest_search_space = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
}

# Define the solver and maximum iterations for the Logistic Regression model
logreg_solver = 'lbfgs'
logreg_max_iter = 30
