# isinstance(variable, type)
# isinstance(stringTest, str)

# This will be used to make the dataset ready to be
# processed.

import ML_base
import numpy as np
import pandas as pd


def dataset_discovery(data):

    # List for each of the feature types
    categorical = []
    numerical = []

    print("Welcome to Data Discovery!")

#    data_type = input("Will this be supervised or unsupervised?")
#    if data_type.lower() == "supervised":
#        classifier = input("What column will the classifier be found?")
#    elif data_type.lower() == "unsupervised":
#        classifier = 999
#    else:
#        print("Invalid selection.")
#        return

    number_of_columns = data.columns
    # Number of features in the dataset

    for every in number_of_columns:
        for each in range(100):
            # for each column, use every row up to 100
            if isinstance(data[every][each], str):
                # If any value within that column is a string, it categorical
                categorical.append(every)
                # Add it to the list then break to the next column
                break
                # If it is a not a string, then it is a number

    # Make a list of the remaining, non-categorical features
    numerical = list(set(number_of_columns) - set(categorical))

    return categorical, numerical
