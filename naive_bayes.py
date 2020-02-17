# Naive Bayes
# The primary functions used to define the Naive Bayes machine learning algorithm
# Called by ML_main

import basic_math
import pickle
from collections import Counter


class naive_bayes:

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

    def save_instance(self):
        with open('./ml_data/nb_instance', 'wb') as save_file:
            pickle.dump(self, save_file)

    # Determines which class a given sample belongs to
    def nb_predict(self):
        for i in range(len(self.data)):
            print("Predicting {:3.2%}".format(i / (len(self.data))), end="\r")
            # Returns the probability of each classification
            self.class_count()
            row = self.data.iloc[i]
            # Probability of a sample belonging to 50000+
            # classProb[0]
            # Probability of a sample belonging to -50000
            # classProb[1]
            for eachValue in range(len(self.data.columns)):
                self.classProb[0] *= basic_math.machine_learning.probability(row[eachValue],
                                                                             self.summaries[eachValue][0],
                                                                             self.summaries[eachValue][1])
                self.classProb[1] *= basic_math.machine_learning.probability(row[eachValue],
                                                                             self.summaries[eachValue][0],
                                                                             self.summaries[eachValue][1])
            if self.classProb[0] > self.classProb[1]:
                self.predictions.append(int(0))
            else:
                self.predictions.append(int(1))

    def class_count(self):
        local_count = Counter(self.classifier)
        # Counts the numbers of 0's and 1's in the classifier list and stores them
        self.classProb[0] = local_count[0]
        self.classProb[1] = local_count[1]

    def main(self):
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

        self.save_instance()
