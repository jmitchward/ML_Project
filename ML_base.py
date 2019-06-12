# Jason Ward 2017-2019

import pandas as pd
from math import sqrt
from math import pi
from math import exp


# NOTE: Is it faster to put testing data and training data in two seperate class instances to eliminate passing
# the dataset being used in a function every time? The weights for LR would have to be passed between after the test
# data is completed, the NB percentages the same. How many functions would I need to rewrite to accommodate? Later.

# NOTE: Should these be static? Does their function depend on the class instance?

# Import the data from the UCI repository, assign it to separate variables
class machine_learning():
    # Declaring here to pull out any specificity for future use
    # These declarations must ONLY be non dependent on class function call

    # 41 total features of the data set, 31 require encoding
    numFeatures = [0, 2, 3, 5, 16, 17, 18, 24, 30, 39]
    catFeatures = [1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 31, 32, 33,
                   34, 35, 36, 37, 38, 40]
    questionFeatures = [25, 26, 27, 29]

    #train_data = pd.read_csv(
    #    'http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz', header=None)
    #test_data = pd.read_csv(
    #    'https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz',
    #    header=None)

#    def __init__(self):
#        # Request dataframes when class is initialized for future use
#        data_pick = input("Which dataset is being used?")
#        if data_pick.lower() == "test":
#            print("Formatting testing data.")
#            self.data = self.test_data
#            self.format_data()
#        elif data_pick.lower() == "train":
#            print("Formatting training data.")
#            self.data = self.train_data
#            self.format_data()
#
#        else:
#            print("Quit fucking around.")
#            self.data = self.train_data
#            self.format_data()

    @staticmethod
    def encode_values(data, column):
        data[column] = data[column].astype('category')
        data[column] = data[column].cat.codes
        data[column] = data[column].astype('int')

    @staticmethod
    def data_data(data):
        print("Information about the dataset is needed to continue. ")
        # Try and glean the data you can, like how many attributes there are
        # Also the data type that they are. Numerical or otherwise.
        # Once that is found out it will be easy to find the rest


    @staticmethod
    def format_data(data):
        # Abstract function definition being avoided for testing purposes
        print('Encoding categorical features.')
        for each in machine_learning.catFeatures:
            machine_learning.encode_values(each)
            # Encode class value.
            # Changes the actual value to a numerical value, allowing it to be used in the algorithm

        machine_learning.encode_values(41)
        for value in machine_learning.questionFeatures:
            data[value].replace(' ?', data.describe(include='all')[value][2], inplace=True)
        # Take the categorical features, feed them to the encode function.
        print('Encoding numerical features.')
        for each in machine_learning.numFeatures:
            mean, std = data[each].mean(), data[each].std()
            data.iloc[:, each] = (data[each] - mean) / std
        # Standardize categorical features
        print('Standardizing categorical features.')
        for each in machine_learning.catFeatures:
            data.loc[:, each] = (data[each] - data[each].mean()) / data[each].std()

    # Calculates probability
    @staticmethod
    def probability(value, mean, sdev):
        condProb = exp(-((value - mean) ** 2 / (2 * sdev ** 2)))
        return (1 / (sqrt(2 * pi) * sdev)) * condProb

    # Calculates the mean and standard deviation for each column
    @staticmethod
    def basic_calc(data):
        summaries = list()

        # for each in hiImpact:
        for each in range(41):
            mean = data[each].mean()
            sdev = data[each].std()
            # creates a list containing the mean and s-dev for each feature in the set.
            summaries.append([mean, sdev])
        return summaries

    @staticmethod
    def log_reg_accuracy(data, predict):
        correct = 0
        print('')
        dSize = len(data)
        for i in range(dSize):
            print("Scoring {:3.2%}".format(i / (len(data))), end="\r")
            if data.iloc[i][41] == predict[i]:
                correct += 1
        print('')
        print('Accuracy:', round(((correct / dSize) * 100.0)), '%')

    @staticmethod
    def accuracy(data, predict):
        # This function does not work without the test data formatted. FIX!
        print('Beginning accuracy rating.')
        correct = 0
        dSize = len(data)
        for i in range(dSize):
            if data.iloc[i][41] == predict[i]:
                correct += 1
        print('Accuracy:', round(((correct / dSize) * 100.0)), '%')


