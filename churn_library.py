# -*- coding: utf-8 -*-
"""
Author:  Thiago Meireles Grabe
Date:    08-July-2021
Repository: https://github.com/ThiagoGrabe/Predict-Customer-Churn

This script provides all functions for churn library. This code is part of an analysis of a Kaggle dataset and more information may be found in the README.md in the git repository.

The module consists of the core code of the churn library:
- Data Import
- Data Preparation
- Exploratory Analysis & Plots
- Encoder for Categorical Columns
- Feature Engineering
- Model Training, Saving and Import
- Model Evaluation & Plots 
"""
import os
import json

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

sns.set()

# Absolute path
ABS_PATH = os.path.dirname(os.path.abspath(__file__))

# Config file (Config.json)
config = open(os.path.join(ABS_PATH, 'config.json'))
config = json.load(config)

# Path from config file
plot_res_folder  = os.path.join(ABS_PATH, config['plot_results_folder'])
plot_eda_folder  = os.path.join(ABS_PATH, config['plot_eda_folder'])
data_folder  = os.path.join(ABS_PATH, config['data_folder'])
model_folder = os.path.join(ABS_PATH, config['model_folder'])
data_file    = config['data_file']

# Categorical Columns and response variable to test
cat_columns     = config['cat_columns']
num_columns     = config['quant_columns']
keep_columns    = config['keep_cols']
response_string = config['response_string']

# Split Dataset parameters
test_size    = float(config['test_size'])
random_state = int(config['random_state'])

# Grid Search Grid
param_grid = config['param_grid']
grid_search_cv = int(config['grid_search_cv'])

# Models names
rfc_model_name    = config['rfc_model']
lgm_model_name    = config['lgm_model']

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def create_churn(df):
    '''
    returns dataframe with a new column named 'churn'

    input:
            df: pandas dataframe
    output:
            df: pandas dataframe
    '''
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df

def plot_eda(df):
    '''
    Save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig(os.path.join(plot_eda_folder, 'churn_hist.png'))
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig(os.path.join(plot_eda_folder, 'Customer_Age_hist.png'))
    plt.figure(figsize=(20, 10))
    df['Marital_Status'].value_counts('normalize').plot(kind='bar')
    plt.savefig(os.path.join(plot_eda_folder, 'Marital_status.png'))
    plt.figure(figsize=(20, 10))
    sns.distplot(df['Total_Trans_Ct'])
    plt.savefig(os.path.join(plot_eda_folder, 'Total_Trans.png'))
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(os.path.join(plot_eda_folder, 'Heatmap.png'))


def perform_eda(df):
    '''
    perform eda on df and return True if not null values were found
    input:
            df: pandas dataframe

    output:
            check: bool
    '''
    return df.notnull().sum().all()


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for category_col in category_lst:
        cat_list = list()
        cat_groups = df.groupby(category_col).mean()[response]
        for val in df[category_col]:
            cat_list.append(cat_groups[val])
        df[str(category_col) + '_' + str(response)] = cat_list
    return df


def perform_feature_engineering(df, response, test_size=0.3, random_state=42):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]
              test_size: size of the dataset used for testing
              random_state: random_state to split the dataset and set a seed
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    X = df[keep_columns]
    y = df[response]
    # train test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def classification_report_image(y_train,
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
    # Clear plot to start new ones
    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    plt.savefig(plot_res_folder + '/Random_Forest_Report.png')
    # Clear plot to start new ones
    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(plot_res_folder + '/Logistic_Regression_Report.png')

def feature_importance_plot(model, X_data, output_pth=plot_res_folder):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Shap Values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar")
    # Saving figure
    plt.savefig(output_pth + '/Shap_Values.png')
    # Clear plot to start new ones
    plt.clf()
    # Feature Importance
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(20,5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth + '/Feature_Importance.png')
    # Clear plot to start new ones
    plt.clf()

def train_models(X_train, X_test, y_train, y_test):
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
    # Training model
    rfc = RandomForestClassifier(random_state=random_state)
    lrc = LogisticRegression(solver='liblinear')

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=grid_search_cv)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # scores
    print('Random Forest Results')
    print('Test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('Train results')
    print(classification_report(y_train, y_train_preds_rf))
    print('logistic regression results')
    print('Test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('Train results')
    print(classification_report(y_train, y_train_preds_lr))


    # Saving ROC Curves for LR models
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.savefig(os.path.join(plot_res_folder, 'Logistic_Regresssion_ROC_Curve.png'))
    # Combined Plots
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(os.path.join(plot_res_folder, 'ROC_Curve.png'))

    # Clear plot to start new ones
    plt.clf()

    # save best model
    joblib.dump(cv_rfc.best_estimator_, os.path.join(model_folder, rfc_model_name))
    joblib.dump(lrc, os.path.join(model_folder, lgm_model_name))

    # loading best model
    rfc_model = joblib.load(os.path.join(model_folder, rfc_model_name))
    joblib.load(os.path.join(model_folder, lgm_model_name))

    # Clear plot to start new ones
    plt.clf()

    feature_importance_plot(model=rfc_model, X_data=X_test, output_pth=plot_res_folder)
    classification_report_image(y_train=y_train.values,
                                y_test=y_test.values,
                                y_train_preds_lr=y_train_preds_lr,
                                y_train_preds_rf=y_train_preds_rf,
                                y_test_preds_lr=y_test_preds_lr,
                                y_test_preds_rf=y_test_preds_rf)

if __name__ == '__main__':
    path = os.path.join(data_folder, data_file)
    raw_df = import_data(pth=path)
    df_churn = create_churn(df=raw_df)
    plot_eda(df_churn)
    perform_eda(df=df_churn)
    df_enconded = encoder_helper(df=df_churn, 
                                category_lst=cat_columns, 
                                response=response_string)

    xtrain, xtest, ytrain, ytest = perform_feature_engineering(df=df_enconded, response=response_string)
    train_models(X_train=xtrain, X_test=xtest, y_train=ytrain, y_test=ytest)
