# Naive Bayes
# The primary functions used to define the Naive Bayes machine learning algorithm
# Called by ML_main

import ML_base
from collections import Counter


class naive_bayes:

    def __init__(self, train_data, test_data, train_class, test_class):
        print("Training data retrieved.")
        self.data = train_data
        self.classifier = train_class
        print("Testing data retrieved.")
        self.test_data = test_data
        self.test_class = test_class
        self.classProb = [0, 0]
        self.above_count = 0
        self.below_count = 0
        self.summaries = []

        self.main()

    # Determines which class a given sample belongs to
    def predict(self, row):
        # Returns the probability of each classification
        self.class_count()
        # Probability of a sample belonging to 50000+
        # classProb[0]
        # Probability of a sample belonging to -50000
        # classProb[1]
        for eachValue in range(len(self.data.columns)):
            self.classProb[0] *= ML_base.machine_learning.probability(row[eachValue], self.summaries[eachValue][0],
                                                                      self.summaries[eachValue][1])
            self.classProb[1] *= ML_base.machine_learning.probability(row[eachValue], self.summaries[eachValue][0],
                                                                      self.summaries[eachValue][1])
        if self.classProb[0] > self.classProb[1]:
            return 0
        else:
            return 1

    def class_count(self):
        local_count = Counter(self.classifier)
        self.classProb[0] = local_count[0]
        self.classProb[1] = local_count[1]

    def main(self):
        predictions = list()
        print("Calculating feature summaries.")
        self.summaries = ML_base.machine_learning.basic_calc(self.data)
        print("Beginning predictions.")
        self.data = self.test_data
        self.classifier = self.test_class
        for i in range(len(self.data)):
            print("Predicting {:3.2%}".format(i / (len(self.data))), end="\r")
            output = self.predict(self.data.iloc[i])
            predictions.append(output)
        print('Determining accuracy.')
        ML_base.machine_learning.accuracy(self.classifier, predictions)
