'''
This module holds all unit tests that are applied to churn_library.py.
Running this file with the prefix "python" creates a comprehensive log file
in logs/churn_library.log. Running this file with the prefix "pytest" results
in running all unit tests.

Author: David Hedderich
Date: 02.03.2023
'''

import os
import logging
import numpy as np
from churn_library import *

logging.basicConfig(
    filename='logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


class TestClass():
    '''
    This class creates a test object of ChurnPredictor for unit tests and logging purposes.

    Attributes:
        None

    Methods:
        - test_import: Testing the data import to the class
        - test_eda: Perform tests regarding the EDA and its resulting images
        - test_enoder_helper: Test if the categorical column encoding works appropriately
        - test_perform_feature_engineering: Test the feature engineering regarding the train/test split
        - test_train_models: Testing if all models are created successfully
        - test_classification_report_image: Test if all evaluation result images are created successfully ftasssss
    '''

    def setup_class(self):
        '''
        setup a test object of the class ChurnPredictor for test purposes
        '''
        self.predictor = ChurnPredictor(
            'data/bank_data.csv',
            'Attrition_Flag',
            'Attrited Customer')

    def test_import(self):
        '''
        test data import - this example is completed for you to assist with the other test functions
        '''

        # Test if the correct file exists in the correct directory
        try:
            self.predictor.import_data()
            logging.info("INFO: Testing import_data: SUCCESS")
        except FileNotFoundError as err:
            logging.error("ERROR: Testing import_data: The file wasn't found")
            raise err

        # Test if the DataFrame is empty
        try:
            assert self.predictor.dataframe.shape[0] > 0
            assert self.predictor.dataframe.shape[1] > 0
            logging.info(
                "INFO: Testing import_data: The file is not empty: SUCCESS")
        except AssertionError as err:
            logging.error(
                "ERROR: Testing import_data: The file doesn't appear to have rows and columns")
            raise err

    def test_eda(self):
        '''
        test perform eda function
        '''

        # Test if the directory images/eda exists
        try:
            file_path = 'images/eda'
            target_directory = os.path.dirname(file_path)
            assert os.path.exists(target_directory) is True
            logging.info(
                "INFO: Testing perform_eda - Target directory exists: SUCCESS")
        except (FileNotFoundError, AssertionError) as err:
            logging.error(
                "ERROR: Testing perform_eda: The target directory wasn't found")
            raise err

        # Test if the images of the EDA were all created successfully
        try:

            for column in self.predictor.cat_columns:
                image_file_path = f'images/eda/{column}_cat_univariate.png'
                current_column = column
                assert os.path.exists(image_file_path) is True
                logging.info("INFO: Testing perform eda - EDA result for column %s saved correctly: SUCCESS" %(column))

            for column in self.predictor.num_columns:
                image_file_path = f'images/eda/{column}_num_univariate.png'
                current_column = column
                assert os.path.exists(image_file_path) is True
                logging.info("INFO: Testing perform eda - EDA result for column %s saved correctly: SUCCESS" %(column))

            image_file_path = 'images/eda/bivariate.png'
            assert os.path.exists(image_file_path) is True
            current_column = 'bivariate analysis'
            logging.info(
                "INFO: Testing perform eda - EDA result for bivariate analysis saved correctly: SUCCESS")

        except (FileNotFoundError, AssertionError) as err:
            logging.error("ERROR: Testing perform_eda: The EDA result for %s wasn't found" %(current_column))
            raise err

    def test_encoder_helper(self):
        '''
        test encoder helper
        '''

        # Check if all categorical mean encodings worked accordingly and
        # created an entry between 0 and 1 for a binary classification problem
        try:
            for column in self.predictor.cat_columns:
                unique_values = self.predictor.dataframe[column].unique()
                within_zero_and_one = np.logical_and(
                    unique_values >= 0, unique_values <= 1)
                assert np.any(np.logical_not(within_zero_and_one)) == False
            logging.info("INFO: Testing encoder_helper: SUCCESS")

        except AssertionError as err:
            logging.error(
                "ERROR: Testing encoder_helper: The mean encoder did not work as intended due to values outside of [0,1]")
            raise err

    def test_perform_feature_engineering(self):
        '''
        test perform_feature_engineering
        '''

        # Check if x_train and x_test do not contain the target column
        try:
            assert (
    			self.predictor.target_column in self.predictor.x_train.columns) is False
            assert (
    			self.predictor.target_column in self.predictor.x_test.columns) is False
            logging.info(
                "INFO: Testing perform_feature_engineering - x_train & x_test do not contain target column: SUCCESS")
        except AssertionError as err:
            logging.error(
                "ERROR: Testing perform_feature_engineering: x_train or x_test still have the target column")
            raise err

        # Check the shape of the four dataframes
        try:
            assert self.predictor.x_test.shape[0] > 0
            assert self.predictor.x_test.shape[1] > 0
            assert self.predictor.x_train.shape[0] > 0
            assert self.predictor.x_train.shape[1] > 0
            assert self.predictor.y_test.shape[0] > 0
            assert self.predictor.y_train.shape[0] > 0
            logging.info(
                "INFO: Testing perform_feature_engineering - No DataFrame for training and testing is empty - SUCCESS")
        except AssertionError as err:
            logging.error(
                "ERROR: Testing perform_feature_engineering: One of the dataframes used for testing is empty")
            raise err

    def test_train_models(self):
        '''
        test train_models
        '''
        # Check if directory ./models exists
        try:
            file_path = './models'
            target_directory = os.path.dirname(file_path)
            assert os.path.exists(target_directory) is True
            logging.info(
                "INFO: Testing train_models - Target directory exists: SUCCESS")

        except (AssertionError, FileNotFoundError) as err:
            logging.error(
                "ERROR: Testing train_models: The target directory wasn't found")
            raise err

        # Check if models were created successfully in ./models/
        try:
            logreg_file_path = './models/logistic_model.pkl'
            rf_file_path = './models/rfc_model.pkl'
            assert os.path.exists(logreg_file_path) is True
            assert os.path.exists(rf_file_path) is True
            logging.info(
                'INFO: Testing train_models - Models were created successfully: SUCCESS')

        except (AssertionError, FileNotFoundError) as err:
            logging.error(
                "ERROR: Testing train_models: One of the models wasn't created successfully")
            raise err

    def test_classification_report_image(self):
        '''
        test classification_report_image
        '''

        # Check if directory ./images/results exists
        try:
            file_path = './images/results'
            target_directory = os.path.dirname(file_path)
            assert os.path.exists(target_directory) is True
            logging.info(
                "INFO: Testing classification_report_image - Target directory exists: SUCCESS")

        except (AssertionError, FileNotFoundError) as err:
            logging.error(
                "ERROR: Testing classification_report_image: The target directory wasn't found")
            raise err

        # Check if images were created successfully in ./images/results
        try:
            roc_path = 'images/results/combined_roc.png'
            shap_path = 'images/results/feature_importance_SHAP.png'
            logreg_path = 'images/results/logistic_model_roc.png'
            rforest_path = 'images/results/rforest_model_roc.png'
            assert os.path.exists(roc_path) is True
            assert os.path.exists(shap_path) is True
            assert os.path.exists(logreg_path) is True
            assert os.path.exists(rforest_path) is True
            logging.info(
                'INFO: Testing classification_report_image - Images were created successfully: SUCCESS')

        except (AssertionError, FileNotFoundError) as err:
            logging.error(
                "ERROR: Testing classification_report_image: One of the images wasn't found and thereby was potentially not created successfully")
            raise err


if __name__ == "__main__":
    tester = TestClass()
    tester.setup_class()
    tester.test_import()
    tester.test_eda()
    tester.test_encoder_helper()
    tester.test_perform_feature_engineering()
    tester.test_train_models()
    tester.test_classification_report_image()
