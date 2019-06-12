# Do import calls as needed
# To be used for the ML foundation
import pandas as pd

import ML_LR
import ML_NB
import ML_base

print("Welcome. The default dataset is loaded. ")


def menu():

    # Pure for testing purposes.
    train_data = pd.read_csv(
        'http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz', header=None)
    test_data = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz',
        header=None)

    print("1. Import database")
    print("2. Format database")
    print("3. Run Logistic Regression")
    print("4. Run Naive Bayes")
    print("5. Run Decision Tree")
    print("6. Run Predictions ")

    choice = input("What would you like to do?")

    if choice == 1 or "import data":
        # Input control please
        TTD = input("Would you like to import the training and test data separately or split automatically?")
        if TTD == "separately":
            url = input("Enter the URL for the training data:")
            train_data = pd.read_csv(url)
            url = input("Enter the URL for the test data:")
            test_data = pd.read_csv(url)
            # Successful and fail checks needed

    elif choice == 2 or "format data":
        choice_format = input("Which dataset would you like to format?")
        if choice_format.lower() == "training data":
            print("Formatting training data.")
            ML_base.machine_learning.format_data(train_data)
        elif choice_format.lower() == "test data":
            print("Formatting test data.")
            ML_base.machine_learning.format_data(test_data)
        else:
            print("Training data or test data?")

    elif choice == 3 or "logistic regression":
        log_reg = ML_LR.logistic_regression.main(train_data, test_data)

#    elif choice == 4 or "decision tree":
#        d_tree = ML_DR.decision_tree.main(train_data, test_data)

    elif choice == 5 or "naive bayes":
        n_bayes = ML_NB.naive_bayes.main(train_data, test_data)

    elif choice == 6 or "predict":
        choice_predict = input("Individual or group prediction?")
        if choice_predict == "individual":
            # Iterate over the dataset features asking for the values
            # of this singular entry
            # THEN
            log_reg.predict()
            # This is calling something that was not defined with this is chosen first
            # While respecting the parameter of the function call
        elif choice_predict == "group":
            # Request either a pre defined new list of entries or manual input


    else:
        print("Invalid.")
