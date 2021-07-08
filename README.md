# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Customer churn is one of the most important metrics for a growing business to evaluate. While it's not the happiest measure, it's a number that can give your company the hard truth about its customer retention.

It's hard to measure success if you don't measure the inevitable failures, too. While you strive for 100% of customers to stick with your company, that's simply unrealistic. That's where customer churn comes in.

This project provides a simple pipeline with clean code to predict wether or not a customer is about to churn in a credit card company.

### What Is Customer Churn?
Customer churn is the percentage of customers that stopped using your company's product or service during a certain time frame. You can calculate churn rate by dividing the number of customers you lost during that time period -- say a quarter -- by the number of customers you had at the beginning of that time period.

For example, if you start your quarter with 400 customers and end with 380, your churn rate is 5% because you lost 5% of your customers.

Obviously, your company should aim for a churn rate that is as close to 0% as possible. In order to do this, your company has to be on top of its churn rate at all times and treat it as a top priority.

### The Dataset 

This dataset is from a [Kaggle Dataset](https://www.kaggle.com/sakshigoyal7/credit-card-customers) and its descriptions explains the given problem:

> A manager at the bank is disturbed with more and more customers leaving their credit card services. They would really appreciate if one could predict for them who is gonna get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction

### Why Is Customer Churn Rate Important?
You may be wondering why it's necessary to calculate churn rate. Naturally, you're going to lose some customers here and there, and 5% doesn't sound too bad, right?

Well, it's important because it costs more to acquire new customers than it does to retain existing customers. In fact, an increase in customer retention of just 5% can create at least a 25% increase in profit. This is because returning customers will likely spend 67% more on your company's products and services. As a result, your company can spend less on the operating costs of having to acquire new customers. You don't need to spend time and money on convincing an existing customer to select your company over competitors because they've already made that decision.

Again, it might seem like a 5% churn rate is solid and healthy. You can still make a vast revenue with that churn rate.


## How to run?
This project consists four folders:

1. data
2. images
3. logs
4. models

Each folder has its function evidentiated by each name. There are two main scripts:

1. `churn_script_logging_and_tests.py`
    This script test the churn library.
2. `churn_library.py`
    Main file with the complete pipeline.

```
ipython script_name.py
```

There are some parameters that may be modified in the file `config.json`.

* plot_eda_folder: Folder to save the exploratory analysis images
* plot_results_folder: Folder to save the results and analysis images
* model_folder: FOlder to save and load models
* rfc_model: Random Forest model name
* lgm_model: Logistic Regression model name
* data_folder: Folder where the data is stored
* data_file: The csv file name 
* cat_columns: Categorical columns in the dataset to test
* quant_columns: Numerical columns in the dataset to test
* keep_cols: Columns to be considered in the modeling step to test
* param_grid: Grid Search parameters for random forest model
* grid_search_cv: Cross Validation size for Grid Search
* response_string: Response Variable to encode
* test_size: Test size to use in sckit-learn `train_test_split`function
* sample_fraction: Sample size to test the pipeline when testing

### Logs

The entire logs can be found in the `logs`folder to track the testing script.

## Results

As a reference, the follwoing results were found

#### Feature Engineering Step
![Heatmap](/images/eda/Heatmap.png)


#### ROC Curve
![ROC Curve](/images/results/ROC_Curve.png)

#### Feature Importance
![Feature Importance](/images/results/Feature_Importance.png)

#### Shap Values
![Shap](/images/results/Shap_Values.png)

#### Model Scores
![Shap](/images/results/Logistic_Regression_Report.png)
![Shap](/images/results/Random_Forest_Report.png)


