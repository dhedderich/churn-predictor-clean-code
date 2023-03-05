# Predict Customer Churn utilizing Clean Code principles

## Project Description
This project creates a customer churn predictor utilizing machine learning to accurately predict the churn potential of customers based on give customer features. This is a binary classification problem. Two different machine learning algorithms are applied and compared for the classification: Logistic Regression & Random Forest. 

The so called **ChurnPredictor** is created as a class having all methods necessary to load the training data, perform an EDA and save its results as images, apply feature engineering on the dataset, train the two machine learning models and evaulate their performance, while saving these results as images. Creating an object of the class ChurnPredictor results in the execution of all before mentioned methods.

Next to a focus on clean code principles according to PEP8 there is a focus on appropriate unit tests and logging. Thereby, every method of **ChurnPredictor** is unit tested.

## Files and data description
Overview of the files and data present in the repository:

### data
The training dataset for the machine learning model is in the data/ directory with the name "bank_data.csv".

The dataset consists of a customer database of a bank, which sees more and more customers churning from their credit card service. The dataset consists of 10,000 customers mentioning their age, salary, marital_status, credit card limit, credit card category, etc., while there are nearly 18 features. The target column is "Attrition Flag" informing about if a customer churned or did not churn in the past.

### images
The images/ directory holds two different types of result images. On the one hand, in the directory images/eda the images of the univariate and bivariate EDA are saved. On the other hand, in the directory images/results the visualizations of the evaluation of the two machine learning models are saved as images, displaying the feature importance of the random forest and various ROC curves.

### logs
The logs/ directory holds the logs created when running the unit tests and logging file "churn_script_logging_and_tests.py".

### models
The /models directory holds the machine learning models in .pkl format, created during the **ChurnPredictor** initialization.

### churn_library.py
In the churn_library python file the **ChurnPredictor** class and its methods are created.

### churn_script_logging_and_testing.py
The churn_script_logging_and_testing python files holds the **TestClass**, which creates an object of the **ChurnPredictor** for testing purposes. Furthermore, all methods of the **ChurnPredictor** have their respective unit tests in this file. Next to unit tests, the logging is also done in this file creating appropriate logs for all tests and methods.

### constants.py
The constants python file holds the hyperparameter definition for the two machine learning models. This way it is easier to change and parameterize the training of the models. The methods are imported into churn_library.py at the beginning.

### requirements.txt
This file holds all third party python libraries necessary to run this project.

## Running Files
Description of how to run the files in this repository

### requirements.txt
To initialize this project you have to install the third party python libraries via:
```bash
pip install -r requirements.txt
```

### churn_library.py
To create a **ChurnPredictor** and thereby creating all images of the EDA and evaluation as well as the two machine learning models you have to run the following:

```bash
python churn_library.py
```

### churn_script_logging_and_tests.py
To run all unit tests and receive the result of the tests and possible errors you have to run:

```bash
pytest churn_script_logging_and_tests.py
```

To create a **TestClass** and all logs of creating a **ChurnPredictor** you have to run:

```bash
python churn_script_logging_and_tests.py
```







