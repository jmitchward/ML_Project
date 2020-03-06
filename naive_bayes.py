# Naive Bayes
# The primary functions used to define the Naive Bayes machine learning algorithm
# Called by ML_main

import basic_math
import program_manager
import pandas as pd
from collections import Counter


class naive_bayes(program_manager.menu):

    def __init__(self, train_data, test_data, train_class, test_class):
        print("Training data retrieved.")
        self.data = train_data
        self.results = 0.0
        self.classifier = train_class
        print("Testing data retrieved.")
        self.test_data = test_data
        self.test_class = test_class
        self.classProb = [0, 0]
        self.above_count = 0
        self.below_count = 0
        self.summaries = []
        self.predictions = list()
        self.main()

    # Determines which class a given sample belongs to
    def nb_predict(self, data=pd.DataFrame({'A': []})):
        # For predicting outside of instance training. If the default value, which is empty, is false then there was
        # a dataframe passed for predicting.
        if not data.empty:
            self.data = data
        self.class_count()
        for i in range(len(self.data)):
            print("{:3.2%}".format(i / (len(self.data))), end="\r")
            row = self.data.iloc[i]
            self.core_predict(row)
            if self.classProb[0] > self.classProb[1]:
                self.predictions.append(int(0))
            else:
                self.predictions.append(int(1))
        #    self.classProb[0] = self.initial_count[0]
        #    self.classProb[1] = self.initial_count[1]
        # After prediction is made using the compound percentage, reset the value to initial.

    def core_predict(self, row):
        # Probability of a sample belonging to 50000+
        # classProb[0]
        # Probability of a sample belonging to -50000
        # classProb[1]
        for this_column in range(len(self.data.columns)):
            self.classProb[0] *= basic_math.machine_learning.probability(row[this_column],
                                                                         self.summaries[this_column][0],
                                                                         self.summaries[this_column][1])
            self.classProb[1] *= basic_math.machine_learning.probability(row[this_column],
                                                                         self.summaries[this_column][0],
                                                                         self.summaries[this_column][1])

    def class_count(self):
        self.initial_count = Counter(self.classifier)
        # Counts the numbers of 0's and 1's in the classifier list and stores them

    def main(self):
        self.predictions.clear()
        print("Calculating feature summaries.")
        self.summaries = basic_math.machine_learning.basic_calc(self.data)
        print("Beginning predictions.")
        # All algorithms within the class run on self.data, reassigning avoids parameters
        self.data = self.test_data
        # All algorithms within the class run on self.classifier
        self.classifier = self.test_class
        self.nb_predict()
        print('Determining accuracy.')
        self.results = basic_math.machine_learning.accuracy(self.classifier, self.predictions)

        return self
