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

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# from sklearn.metrics import plot_roc_curve, classification_report

os.environ['QT_QPA_PLATFORM']='offscreen'

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


        def perform_feature_engineering(self, df, response):
                '''
                input:
                        self.dataframe: (DataFrame) pandas dataframe
                        
                output:
                        X_train: X training data
                        X_test: X testing data
                        y_train: y training data
                        y_test: y testing data
                '''

        def classification_report_image(self,
                                        y_train,
                                        y_test,
                                        y_train_preds_lr,
                                        y_train_preds_rf,
                                        y_test_preds_lr,
                                        y_test_preds_rf):
                '''
                produces classification report for training and testing results and stores report as image
                in images folder
                input:
                        y_train: training response values
                        y_test:  test response values
                        y_train_preds_lr: training predictions from logistic regression
                        y_train_preds_rf: training predictions from random forest
                        y_test_preds_lr: test predictions from logistic regression
                        y_test_preds_rf: test predictions from random forest

                output:
                        None
                '''
                pass


        def feature_importance_plot(self, model, X_data, output_pth):
                '''
                creates and stores the feature importances in pth
                input:
                        model: model object containing feature_importances_
                        X_data: pandas dataframe of X values
                        output_pth: path to store the figure

                output:
                        None
                '''
                pass

        def train_models(self, X_train, X_test, y_train, y_test):
                '''
                train, store model results: images + scores, and store models
                input:
                        X_train: X training data
                        X_test: X testing data
                        y_train: y training data
                        y_test: y testing data
                output:
                        None
                '''
                pass

if __name__ == '__main__':
        predictor = ChurnPredictor('data/bank_data.csv', 'Attrition_Flag', 'Attrited_Customer')
        predictor.import_data()
        predictor.perform_eda()
        predictor.encoder_helper()
