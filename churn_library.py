# library doc string
"""
Module that defines all necessary functions as part of a class, that can be used to analyze the data provided in /data, perform some EDA on the data and save the resulting images to /images and trains a machine learning model, that is saved in /models. 


Author: David Hedderich
Date: 26.02.2023
"""

# import libraries
import os
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from constants import random_forest_search_space

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report


class ChurnPredictor():
        """
        The following class imports data from ./data, performs an EDA, saves the EDA results to a folder, encodes specific columns, performs feature engineering, trains a machine learning model, saves a training report to ./images/results and saves the machine learning models to ./models
                Attributes:
                
                Methods:

        """


        def __init__(self, path, target_column, target_column_churn_name):
                """     
                initializes the class ChurnPredictor and assigns the input parameters

                input:
                        path: (str) a path to the csv
                        target_column: (str) column that holds the binary classification resulting in either "Churn" (1) or "No Churn (0)
                        target_column_churn_name: (str) name of the positive class within the binary classification target column that represents the "Churn" (1), if the target column is categorical          
                """
                self.path = path
                self.target_column = target_column
                self.target_column_churn_name = target_column_churn_name

                #TODO: Add all self.function() here for testing to execute all following steps in the initialization of the created object


        def import_data(self):
                """     returns dataframe for the csv found at pth

                input:
                        self.path: (str) a path to the csv
                output:
                        self.dataframe: (DataFrame) pandas dataframe
                """	
                
                # Read in .csv file
                self.dataframe = pd.read_csv(self.path)

                # Encode target column to create a binary classification problem
                self.dataframe[self.target_column] = self.dataframe[self.target_column].apply(lambda val: 1 if val == self.target_column_churn_name else 0)

                return self.dataframe


        def perform_eda(self):
                '''
                perform eda on df and save figures to images folder
                input:
                        self.dataframe: (DataFrame) pandas dataframe

                output:
                        None
                '''
                
                # Print out some overview metrics regarding the dataset
                print("SUMMARY of the imported data: \n")
                print("Shape: \n")
                print(self.dataframe.shape)
                print("Number of NaN values: \n")
                print(self.dataframe.isnull().sum())
                print("Statistics of numerical columns: \n")
                print(self.dataframe.describe())

                # Identify column types
                self.cat_columns = self.dataframe.select_dtypes(include = 'object').columns
                self.num_columns = self.dataframe.select_dtypes(include = ['int', 'float']).columns

                # Plot and save univariate analysis and bivariate analysis
                for column in self.num_columns:
                        fig = plt.figure(figsize=(20,10))
                        self.dataframe[column].hist()
                        fig.savefig('images/eda/{}_num_univariate.png'.format(column))
                        plt.close(fig)
        
                for column in self.cat_columns:
                        fig = plt.figure(figsize=(20,10))
                        self.dataframe[column].value_counts('normalize').plot(kind='bar')
                        fig.savefig('images/eda/{}_cat_univariate.png'.format(column))
                        plt.close(fig)

                fig = plt.figure(figsize=(20,10)) 
                sns.heatmap(self.dataframe.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
                fig.savefig('images/eda/bivariate.png'.format(column))
                plt.close(fig)


        def encoder_helper(self):
                '''
                helper function to turn each categorical column into a new column with
                propotion of churn for each category - associated with cell 15 from the notebook

                input:
                        self.dataframe: (DataFrame) pandas dataframe
                        self.cat_columns: (list) list of columns that contain categorical features
                        
                output:
                        self.dataframe: (DataFrame) pandas dataframe with new columns
                '''
                
                for column in self.cat_columns:
                        helper_lst = []
                        cat_column_groups = self.dataframe.groupby(column).mean()[self.target_column]

                        for val in self.dataframe[column]:
                                helper_lst.append(cat_column_groups.loc[val])

                        self.dataframe[column] = helper_lst

                return self.dataframe


        def perform_feature_engineering(self):
                '''
                input:
                        self.dataframe: (DataFrame) pandas dataframe
                        
                output:
                        self.X_train: X training data
                        self.X_test: X testing data
                        self.y_train: y training data
                        self.y_test: y testing data
                '''

                df_train_columns = self.dataframe.drop([self.target_column], axis=1)              
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df_train_columns, self.dataframe[self.target_column], test_size= 0.3, random_state=42)
                return  self.X_train, self.X_test, self.y_train, self.y_test


        def train_models(self):
                '''
                train, store model results: images + scores, and store models
                input:
                        self.X_train: X training data
                        self.X_test: X testing data
                        self.y_train: y training data
                        self.y_test: y testing data
                output:
                        None
                '''
                # grid search
                rfc = RandomForestClassifier(random_state=42)
                lrc = LogisticRegression(solver='lbfgs', max_iter=30) #TODO: increase to 3000 after testing
                
                # Define search space for Random Forest
                param_grid = random_forest_search_space

                cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
                cv_rfc.fit(self.X_train, self.y_train)
                print(self.X_train.head()) #TODO: remove after testing
                print(self.y_train.head()) #TODO: remove after testing
                lrc.fit(self.X_train, self.y_train)

                self.y_train_preds_rf = cv_rfc.best_estimator_.predict(self.X_train)
                self.y_test_preds_rf = cv_rfc.best_estimator_.predict(self.X_test)

                self.y_train_preds_lr = lrc.predict(self.X_train)
                self.y_test_preds_lr = lrc.predict(self.X_test)      

                 # save best model
                joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
                joblib.dump(lrc, './models/logistic_model.pkl')

        def classification_report_image(self):
                '''
                produces classification report for training and testing results and stores report as image
                in images folder
                input:
                        self.y_train: training response values
                        self.y_test:  test response values
                        y_train_preds_lr: training predictions from logistic regression
                        y_train_preds_rf: training predictions from random forest
                        y_test_preds_lr: test predictions from logistic regression
                        y_test_preds_rf: test predictions from random forest

                output:
                        None
                '''
                # Print scores to terminal
                print('random forest results')
                print('test results')
                print(classification_report(self.y_test, self.y_test_preds_rf))
                print('train results')
                print(classification_report(self.y_train, self.y_train_preds_rf))

                print('logistic regression results')
                print('test results')
                print(classification_report(self.y_test, self.y_test_preds_lr))
                print('train results')
                print(classification_report(self.y_train, self.y_train_preds_lr))

                # Load models
                rfc_model = joblib.load('./models/rfc_model.pkl')
                lr_model = joblib.load('./models/logistic_model.pkl')   

                # Create and save ROC curves
                lrc_plot = plot_roc_curve(lr_model, self.X_test, self.y_test)
                lrc_plot.figure_.savefig('images/results/logistic_model_roc.png')

                rfc_plot = plot_roc_curve(rfc_model, self.X_test, self.y_test)
                rfc_plot.figure_.savefig('images/results/rforest_model_roc.png')

                combined_figure = plt.figure(figsize=(15, 8))
                axis = plt.gca()
                rfc_disp = plot_roc_curve(rfc_model, self.X_test, self.y_test, ax=axis, alpha=0.8)
                lrc_plot.plot(ax=axis, alpha=0.8)
                combined_figure.savefig('images/results/combined_roc.png')
                plt.close(combined_figure)


        def feature_importance_plot(self):
                '''
                creates and stores the feature importances in pth
                input:
                        model: model object containing feature_importances_
                        self.dataframe: pandas dataframe of X & y values                        

                output:
                        None
                '''
                # Load random forest model
                rfc_model = joblib.load('./models/rfc_model.pkl')

                explainer = shap.TreeExplainer(rfc_model)
                shap_values = explainer.shap_values(self.X_test)
                shap.summary_plot(shap_values, self.X_test, plot_type="bar", show=False)
                plt.savefig('images/results/feature_importance_SHAP')
        

if __name__ == '__main__':
        predictor = ChurnPredictor('data/bank_data.csv', 'Attrition_Flag', 'Attrited Customer')
        predictor.import_data()
        predictor.perform_eda()
        predictor.encoder_helper()
        predictor.perform_feature_engineering()
        predictor.train_models()
        predictor.classification_report_image()
        predictor.feature_importance_plot()