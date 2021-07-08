# -*- coding: utf-8 -*-
"""
Author:  Thiago Meireles Grabe
Date:    08-July-2021
Repository: https://github.com/ThiagoGrabe/Predict-Customer-Churn

This script test all functions from 'churn_library.py' in order to keep the library well designed and achieving high code quality. This code is part of an analysis of a Kaggle dataset and more information may be found in the README.md in the git repository.

The module test the following functions in churn library:
- Data Import
- Data Preparation
- Exploratory Analysis & Plots
- Encoder for Categorical Columns
- Feature Engineering
- Model Training, Saving and Import
- Model Evaluation & Plots 
"""
from copy import Error
import os
import json
import logging
import pandas as pd
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s. - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Absolute path
ABS_PATH = os.path.dirname(os.path.abspath(__file__))

# Config file (Config.json)
config = open(os.path.join(ABS_PATH, 'config.json'))
config = json.load(config)

# Path from config file
plot_res_folder  = os.path.join(ABS_PATH, config['plot_results_folder'])
plot_eda_folder  = os.path.join(ABS_PATH, config['plot_results_folder'])
data_folder  = os.path.join(ABS_PATH, config['data_folder'])
model_folder = os.path.join(ABS_PATH, config['model_folder'])
data_file    = config['data_file']

# Categorical Columns and response variable to test
cat_columns = config['cat_columns']
num_columns = config['quant_columns']
keep_columns = config['keep_cols']
response_string = config['response_string']

# Split Dataset parameters
test_size = float(config['test_size'])
random_state = int(config['random_state'])
sample_fraction = float(config['sample_fraction'])

def test_import(import_data):
    '''
    test import_data function from churn module
    input:
            import_data: python function

    output:
            None
    '''
    try:
        df = import_data(os.path.join(data_folder, data_file))
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

def test_create_churn(create_churn):
    '''
    test create_churn function from churn module
    input:
            import_data: python function

    output:
            None
    '''
    df = cls.import_data(os.path.join(data_folder, data_file))
    try:
        assert isinstance(df, pd.DataFrame)
        logging.info("DataFrame type: SUCCESS")
    except AssertionError as err:
        logging.error("DataFrame type: variable is not a pandas DataFrame")
        raise err
    try:
        assert set(['Attrition_Flag']).issubset(df.columns)
        logging.info("Attrition_Flag column: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Attrition_Flag column: Attrition_Flag column was not found.")
        raise err
    try:
        df = create_churn(df)
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("DataFrame returned from create_churn function: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Check Dataset Fail: The create_churn function does not return correctly")
        raise err
    try:
        assert set(['Churn']).issubset(df.columns)
        logging.info("Churn column: SUCCESS")
    except AssertionError as err:
        logging.error("Churn column: Churn column was not found.")
        raise err


def test_plot_eda(plot_eda):
    '''
    test plot_eda function from churn module
    input:
            import_data: python function

    output:
            None
    '''
    df = cls.import_data(os.path.join(data_folder, data_file))
    df = cls.create_churn(df)
    try:
        assert set(['Churn', 'Customer_Age', 'Marital_Status',
                   'Total_Trans_Ct']).issubset(df.columns)
        logging.info("Plot columns: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Plot columns: Plot Columns column was not found. Possible Columns: " + str(df.columns))
        raise err

    try:
        plot_eda(df=df)
        logging.info("Plot EDA: SUCCESS")
    except Error as err:
        logging.error("Plot EDA: Fail to plot exploratory analysis images.")
        raise err

def test_eda(perform_eda):
    '''
    test perform_eda function from churn module
    input:
            import_data: python function

    output:
            None
    '''
    df = cls.import_data(os.path.join(data_folder, data_file))
    df = cls.create_churn(df)
    try:
        if perform_eda(df):
            logging.info("EDA Analysis with not null values: SUCCESS")
        else:
            logging.info("EDA Analysis with some null values: SUCCESS")
    except Error as err:
        logging.error("Testing  EDA on dataframe: Fail to perform EDA")
        raise err

def test_encoder_helper(encoder_helper):
    '''
    test encoder_helper function from churn module
    input:
            import_data: python function

    output:
            None
    '''
    df = cls.import_data(os.path.join(data_folder, data_file))
    df = cls.create_churn(df)

    try:
        assert set(cat_columns).issubset(df.columns)
        logging.info("Categorical columns is in DataFrame: SUCCESS")
    except AssertionError as err:
        logging.error("Categorical columns is not in DataFrame. Categorical Columns: " +
                      str(df.select_dtypes(include=['category']).columns))
        raise err
    try:
        assert set(num_columns).issubset(df.columns)
        logging.info("Numerical columns is in DataFrame: SUCCESS")
    except AssertionError as err:
        logging.error("Numerical columns is not in DataFrame. Numerical Columns: " +
                      str(df.select_dtypes(include=['number']).columns))
        raise err
    try:
        assert set([response_string]).issubset(df.columns)
        logging.info("Response columns is in DataFrame: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Response column is not in DataFrame. Possible numerical Columns: " + str(
                df.select_dtypes(
                    include=['number']).columns))
        raise err
    try:
        encoder_helper(
            df=df,
            category_lst=cat_columns,
            response=response_string)
        logging.info("Encoder Dataframe: SUCCESS")
    except Error as err:
        logging.error("Encoder Dataframe: Fail to encoder categorical columns")
        raise err

def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature _engineering function from churn module
    input:
            import_data: python function

    output:
            None
    '''
    df = cls.import_data(os.path.join(data_folder, data_file))
    df = cls.create_churn(df)
    df = cls.encoder_helper(
        df,
        category_lst=cat_columns,
        response=response_string)

    try:
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            df=df, response=response_string, test_size=test_size, random_state=random_state)
        logging.info("Perform Feature Engineering: SUCCESS")
    except Error as err:
        logging.error("Perform Feature Engineering failed!")
        raise err

    try:
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
        logging.info("X_train split: SUCCESS")
    except AssertionError as err:
        logging.error("X_train Failed")
        raise err

    try:
        assert y_train.shape[0] > 0
        logging.info("y_train split: SUCCESS")
    except AssertionError as err:
        logging.error("y_train Failed")
        raise err

    try:
        assert X_test.shape[0] > 0
        assert X_test.shape[1] > 0
        logging.info("X_test split: SUCCESS")
    except AssertionError as err:
        logging.error("X_test Failed")
        raise err

    try:
        assert y_test.shape[0] > 0
        logging.info("y_test split: SUCCESS")
    except AssertionError as err:
        logging.error("y_test Failed")
        raise err

def test_train_models(train_models):
    '''
    test train_models function from churn module
    input:
            import_data: python function

    output:
            None
    '''
    df = cls.import_data(os.path.join(data_folder, data_file))
    df = cls.create_churn(df)
    df = cls.encoder_helper(
        df,
        category_lst=cat_columns,
        response=response_string)

    # Sampling x% of the dataset for testing (we do not need the complete
    # dataset to test)
    df = df.sample(frac=sample_fraction)
    logging.info("Sampling " + str(round(sample_fraction * 100, 2)
                                   ) + " percent of the dataset for testing.")

    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        df=df, response=response_string, test_size=test_size, random_state=random_state)

    try:
        train_models(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test)
        logging.info("Training Models test: SUCCESS")
    except Error as err:
        logging.error("Training Models test failed!")
        raise err

if __name__ == "__main__":
    test_import(cls.import_data)
    test_create_churn(cls.create_churn)
    test_plot_eda(cls.plot_eda)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
