# Jason Ward 2017-2019

from math import exp
import basic_math
import pandas
import logging


# Its iloc[row][column]
# Features are coluns, datapoints are rows


class logistic_regression:

    def __init__(self, train_data, test_data, train_class, test_class):
        logging.basicConfig(level=logging.DEBUG, format='\n %(asctime)s - %(levelname)s - %(message)s')

        self.weights = [0.0] * (len(train_data.columns) + 1)
        # self.weight_key = 0.0
        self.results = 0.0
        self.predictions = list()

        self.test_data = test_data
        self.test_class = test_class
        self.data = train_data
        self.train_class = train_class
        self.classifier = train_class
        self.lr_intent = "weight"
        self.main()

    def lr_train(self, learn, iterations):
        for eachIter in range(iterations):
            sumError = 0
            for row in self.data.itertuples():
                print("Updating Weights {:3.2%}".format(row[0] / (len(self.data))), end="\r")
                result = self.lr_predict(row)
                error = (result - row[41])
                sumError += .5 * (error ** 2)
                self.weights[0] = self.weights[0] - learn * (1 / (len(self.data))) * error * result
                for i in range(len(self.data.columns)):
                    self.weights[i + 1] = self.weights[i + 1] - learn * (1 / (len(self.data))) * error * row[i]
        return self.weights

    def lr_predict(self, row):
        # Store the initial weight
        weight = self.weights[0]
        for each in range(len(self.data.columns)):
            # Using the logistic function, calculate the predicted output
            weight += (self.weights[each + 1] * row[each])
        if weight < 0:
            return 1.0 - 1 / (1.0 + exp(weight))
        else:
            return 1.0 / (1.0 + exp(-weight))

    def main(self):
        learningRate = 0.2
        iterations = 3
        # Returns the list of weights
        print("Calculating weights.")
        self.weights = self.lr_train(learningRate, iterations)
        logging.debug('self.weights returned')
        self.predictions = list()
        logging.debug('self.predictions reset')
        # Training phrase is complete, redefine working dataset as the test data
        self.data = self.test_data
        logging.debug('self.data redefined to self.test_data')
        self.classifier = self.test_class
        logging.debug('self.classifier defined to self.test_class')
        logging.debug('lr_predict called recalled for self.test_data as self.data')
        for row in self.data.itertuples():
            prediction = self.lr_predict(row)
            if prediction > 0.5:
                self.predictions.append(1)
            else:
                self.predictions.append(0)
        logging.debug('Object left for accuracy rating')
        self.results = basic_math.machine_learning.accuracy(self.test_class, self.predictions)

        return self
