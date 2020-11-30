# Naive Bayes
# The primary functions used to define the Naive Bayes machine learning algorithm
# Called by ML_main

from static_scripts import basic_math
import pandas as pd
import logging
from math import sqrt
from math import pi
from math import exp
from collections import Counter


class naive_bayes:

    def __init__(self, train_data, test_data, train_class, test_class):
        logging.basicConfig(level=logging.DEBUG, format='\n %(asctime)s - %(levelname)s - %(message)s')

        print("Training data retrieved.")
        self.data = train_data
        self.results = 0.0
        self.classifier = train_class
        print("Testing data retrieved.")
        self.initial_count = Counter(self.classifier)
        self.test_data = test_data
        self.test_class = test_class
        self.classProb = [0.0, 0.0]
        self.summaries = []
        self.predictions = list()
        self.main()

    # Determines which class a given sample belongs to
    def nb_predict(self, data=pd.DataFrame({'A': []})):
        logging.debug('nb_predict entered')
        # For predicting outside of instance training. If the default value, which is empty, is false then there was
        # a dataframe passed for predicting.
        # if not data.empty:
        #            self.data = data
        logging.debug('core_predict called')
        self.core_predict()

    #    self.classProb[0] = self.initial_count[0]
    #    self.classProb[1] = self.initial_count[1]
    # After prediction is made using the compound percentage, reset the value to initial.

    def core_predict(self):
        # Model assumes normal distribution
        logging.debug('core_predict entered')
        for each_row in range(len(self.data)):
            self.class_count()
            current_row = self.data.iloc[each_row]
            self.class_count()
            for each_column in range(len(self.data.columns)):
                current_row_column = current_row[each_column]
                step_one = (current_row_column - self.summaries[each_column][0]) ** 2
                step_two = 2 * (self.summaries[each_column][1] ** 2)
                step_three = exp(-(step_one/step_two))
                step_four = 1 / (sqrt(2 * pi) * self.summaries[each_column][1])
                step_five = step_four * step_three
            # classProb holds the initial probability already
                self.classProb[0] *= step_five
                self.classProb[1] *= step_five
            if self.classProb[0] > self.classProb[1]:
                self.predictions.append(int(0))
            else:
                self.predictions.append(int(1))
        # Probability of a sample belonging to 50000+
        # classProb[0]
        # Probability of a sample belonging to -50000
        # classProb[1]

    def class_count(self):
        self.classProb[0] = self.initial_count[0] / len(self.data)
        self.classProb[1] = self.initial_count[1] / len(self.data)
        # Counts the numbers of 0's and 1's in the classifier list and stores them

    def main(self):
        print("Calculating feature summaries.")
        self.summaries = basic_math.machine_learning.basic_calc(self.data)
        self.class_count()
        print("Beginning predictions.")
        # All algorithms within the class run on self.data, reassigning avoids parameters
        self.data = self.test_data
        # All algorithms within the class run on self.classifier
        self.classifier = self.test_class
        self.nb_predict()
        print('Determining accuracy.')
        self.results = basic_math.machine_learning.accuracy(self.classifier, self.predictions)

        return self
