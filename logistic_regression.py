# Jason Ward 2017-2019

from math import exp
import pickle
import basic_math


# Its iloc[row][column]
# Features are coluns, datapoints are rows


class logistic_regression:

    def __init__(self, train_data, test_data, train_class, test_class):
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
        # Gradient descent is only used to establish weights, which are only established using
        # training data, therefore it will only ever need receive training data.
        # WEIGHT CALCULATION
        for eachIter in range(iterations):
            sumError = 0
            print("Beginning ", eachIter, " of ", iterations, " iterations.")
            for datapoints in range(len(self.data)):
                error = 0
                # There are 40 weights, one for each individual column
                # Each weight is built from the sum of the columns
                print("Updating Weights {:3.2%}".format(datapoints / (len(self.data))), end="\r")
                current_row = self.data.iloc[datapoints]
                # PREDICTOR
                weight_key = self.weights[0]
                # Store the initial weight
                for current_column in range(len(self.data.columns)):
                    # Grab the feature values passed in the single entry given to the function one by one
                    row_value = current_row[current_column]
                    # Get the sum of each value multiplied by the weight of each column in the current row
                    weight_key += (self.weights[current_column] * row_value)
                    # Calculate the second using the second part
                    if weight_key < 0:
                        result = (1.0 - 1 / (1.0 + exp(self.weights[0])))
                    else:
                        result = (1.0 / (1.0 + exp(-self.weights[0])))
                    error = (result - self.classifier[datapoints])
                    sumError += .5 * (error ** 2)
                    self.weights[0] = self.weights[0] - learn * (1 / (len(self.data))) * sumError * result
                    self.weights[current_column+1] = self.weights[current_column+1] - learn * (1 / (len(self.data))) * sumError * current_row[current_column]
        return self.weights

    def lr_predict(self):
        for entries in range(len(self.data)):
            self.prediction = 0.0
            print("Updating Weights {:3.2%}".format(entries / (len(self.data))), end="\r")
            current_entry = self.data.iloc[entries]
            weight_key = self.weights[0]
            for current_feature in range(len(self.data.columns)):
                # Grab the feature values passed in the single entry given to the function one by one
                row_value = current_entry[current_feature]
                # Multiply the weight by the actual value of each row in the feature
                weight_key += (self.weights[current_feature] * row_value)
                self.prediction = (1.0 / (1.0 + exp(-self.weights[0])))
            # Rounds prediction result to 2 decimal places.
            if self.prediction > 0.5:
                self.predictions.append(1)
            else:
                self.predictions.append(0)

    def main(self):
        learningRate = 0.2
        iterations = 3
        # Returns the list of weights
        print("Calculating weights.")
        self.weights = self.lr_train(learningRate, iterations)
        self.predictions = list()
        # Training phrase is complete, redefine working dataset as the test data
        self.data = self.test_data
        self.classifier = self.test_class
        self.lr_predict()

        self.results = basic_math.machine_learning.accuracy(self.test_class, self.predictions)

        return self
