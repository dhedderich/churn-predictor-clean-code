'''

'''

# Define your hyperparameter search space for the random forrest below
random_forest_search_space = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
}