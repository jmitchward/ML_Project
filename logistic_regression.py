# Jason Ward 2017-2019

from math import exp
import pickle
import basic_math


# Its iloc[row][column]
# Features are coluns, datapoints are rows


class logistic_regression:

    def __init__(self, train_data, test_data, train_class, test_class):
        # self.weight_key = 0.0
        self.weights = [0.0] * 41
        self.results = 0.0
        self.predictions = list()

        self.test_data = test_data
        self.test_class = test_class
        self.data = train_data
        self.train_class = train_class

        self.classifier = train_class
        self.main()

    def lr_predict(self, learn, iterations):
        # Gradient descent is only used to establish weights, which are only established using
        # training data, therefore it will only ever need receive training data.
        for eachIter in range(iterations):
            sumError = 0
            for datapoints in range(len(self.data)):
                # There are 40 weights, one for each individual column
                # Each weight is built from the sum of the columns
                print("Updating Weights {:3.2%}".format(datapoints / (len(self.data))), end="\r")
                current_row = self.data.iloc[datapoints]
                weight_key = self.weights[0]
                # Store the initial weight
                for current_column in range(len(self.data.columns)):
                    # Grab the feature values passed in the single entry given to the function one by one
                    row_value = current_row[current_column]
                    # Multiply the weight by the actual value of each row in the feature
                    weight_key += (self.weights[current_column] * row_value)
                if weight_key < 0:
                    result = (1.0 - 1 / (1.0 + exp(self.weights[0])))
                else:
                    result = (1.0 / (1.0 + exp(-self.weights[0])))
                error = (result - self.classifier[datapoints])
                sumError += .5 * (error ** 2)
                self.weights[0] = self.weights[0] - learn * (1 / (len(self.data))) * error * result
                # For each feature, use the LR algorithm to train the weight
                for i in range(len(self.data.columns) - 1):
                    next_value = self.data.iloc[datapoints][i]
                    self.weights[i + 1] = self.weights[i + 1] - learn * (1 / (len(self.data))) * error * next_value
        return self.weights

    def main(self):
        learningRate = 0.2
        iterations = 5
        # Returns the list of weights
        print("Calculating weights.")
        self.weights = self.weight_calculator(learningRate, iterations)
        predictions = list()
        # Training phrase is complete, redefine working dataset as the test data
        self.data = self.test_data
        self.classifier = self.test_class
        for row in range(len(self.data)):
            print("Predicting {:3.2%}".format(row / (len(self.data))), end="\r")
            nextRow = self.data.iloc[row]
            prediction = self.lr_predict(nextRow)
            # Rounds prediction result to 2 decimal places.
            if prediction > 0.5:
                self.predictions.append(1)
            else:
                self.predictions.append(0)

        self.results = basic_math.machine_learning.accuracy(self.test_class, self.predictions)

        return self
