'''

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
logreg_max_iter = 70
