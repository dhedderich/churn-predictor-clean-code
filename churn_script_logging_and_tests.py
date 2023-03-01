import os
import logging
import numpy as np
from churn_library import *

logging.basicConfig(
    filename='logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

class TestClass():

	def setup_class(self):
		'''
		setup a test object of the class ChurnPredictor for test purposes
		'''
		self.predictor = ChurnPredictor('data/bank_data.csv', 'Attrition_Flag', 'Attrited Customer')

	def test_import(self):
		'''
		test data import - this example is completed for you to assist with the other test functions
		'''		

		# Test if the correct file exists in the correct directory
		try:
			self.df = self.predictor.import_data()
			logging.info("INFO: Testing import_data: SUCCESS")
		except FileNotFoundError as err:
			logging.error("ERROR: Testing import_data: The file wasn't found")
			raise err

		try:
			assert df.shape[0] > 0
			assert df.shape[1] > 0
		except AssertionError as err:
			logging.error("ERROR: Testing import_data: The file doesn't appear to have rows and columns")
			raise err


	def test_eda(self):
		'''
		test perform eda function
		'''

		try:
			file_path = './images/eda'
			target_directory = os.path.dirname(file_path)
			assert os.path.exist(target_directory) = True
			logging.info("INFO: Testing perform_eda: SUCCESS")
		except FileNotFoundError as err:
			logging.error("ERROR: Testing perform_eda: The target directory wasn't found")
			raise err
	
		try:
			cat_columns = self.df.select_dtypes(include='object').columns
			num_columns = self.df.select_dtypes(
				include=['int', 'float']).columns
			
			for column in cat_columns:
				image_file_path = f'images/eda/{column}_cat_univariate.png'
				current_column = column
				assert os.path.exists(image_file_path) = True
				logging.info(f"INFO: Testing perform eda - EDA result for column {column} saved correctly: SUCCESS")

			for column in num_columns:
				image_file_path = f'images/eda/{column}_num_univariate.png'
				current_column = column
				assert os.path.exists(image_file_path) = True
				logging.info(f"INFO: Testing perform eda - EDA result for column {column} saved correctly: SUCCESS")

		except FileNotFoundError as err:
				logging.error(f"ERROR: Testing perform_eda: The EDA result for column {current_column} wasn't found")
				raise err


	def test_encoder_helper(self):
		'''
		test encoder helper
		'''
		
		# Check if all mean encodings worked accordingly and created an entry between 0 and 1 for a binary classification problem
		try:
			for column in cat_columns:
				unique_values = self.df[column].unique()
				within_zero_and_one = np.logical_and(unique_values >= 0, unique_values <= 1)
				assert np.any(np.logical_not(between_zero_and_one)) = False
			logging.info("INFO: Testing encoder_helper: SUCCESS")
		
		except AssertionError as err:
			logging.error("ERROR: Testing encoder_helper: The mean encoder did not work as intended due to values outside of [0,1]")
			raise err


	def test_perform_feature_engineering(self):
		'''
		test perform_feature_engineering
		'''


	def test_train_models(self):
		'''
		test train_models
		'''
		# Check if directory ./models exists
		try:
			file_path = './models'
			target_directory = os.path.dirname(file_path)
			assert os.path.exist(target_directory) = True
			logging.info("INFO: Testing train_models - Target directory exists: SUCCESS")
		except FileNotFoundError as err:
			logging.error("ERROR: Testing train_models: The target directory wasn't found")
			raise err

		# Check if models were created successfully in ./models/
		try:
			logreg_file_path = './models/logistic_model.pkl'
			rf_file_path = './models/rfc_model.pkl'
			assert os.path.exists(logreg_file_path) = True
			assert os.path.exists(rf_file_path) = True
			logging.info('INFO: Testing train_models - Models were created successfully: SUCCESS')
		
		except FileNotFoundError as err:
			logging.error(f"ERROR: Testing train_models: One of the models wasn't found and thereby potentially not created successfully")
			raise err

if __name__ == "__main__":
	tester = TestClass()
	tester.setup_class()
	tester.test_import()
	tester.test_eda()
	tester.test_encoder_helper()
	tester.test_perform_feature_engineering()
	tester.test_train_models()









