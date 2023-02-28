import os
import logging
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

		try:
			df = self.predictor.import_data()
			logging.info("Testing import_data: SUCCESS")
		except FileNotFoundError as err:
			logging.error("Testing import_eda: The file wasn't found")
			raise err

		try:
			assert df.shape[0] > 0
			assert df.shape[1] > 0
		except AssertionError as err:
			logging.error("Testing import_data: The file doesn't appear to have rows and columns")
			raise err


	def test_eda(self):
		'''
		test perform eda function
		'''


	def test_encoder_helper(self):
		'''
		test encoder helper
		'''


	def test_perform_feature_engineering(self):
		'''
		test perform_feature_engineering
		'''


	def test_train_models(self):
		'''
		test train_models
		'''


if __name__ == "__main__":
	tester = TestClass()
	tester.setup_class()
	tester.test_import()
	tester.test_eda()
	tester.test_encoder_helper()
	tester.test_perform_feature_engineering()
	tester.test_train_models()









