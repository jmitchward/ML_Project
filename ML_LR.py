# Jason Ward 2017-2019

from math import exp
import pickle
import ML_base


class logistic_regression(ML_base.machine_learning):
    # Returns list of 41 zeros
    weights = [0.0] * 41

    def __init__(self, train_data, test_data):
        print("Training data retrieved.")
        self.train_data = train_data
        print("Test data retrieved.")
        self.test_data = test_data
        # This would be an excellent point to make sure the data is encoded
        self.main()

    def gradient_descent(self, learn, iterations):
        # Gradient descent is only used to establish weights, which are only established using
        # training data, therefore it will only ever need receive training data.
        for eachIter in range(iterations):
            sumError = 0
            for row in range(len(self.train_data)):
                print("Updating Weights {:3.2%}".format(row / (len(self.train_data))), end="\r")
                row = self.train_data.iloc[row]
                result = self.predict(row)
                error = (result - row[41])
                sumError += .5 * (error ** 2)
                self.weights[0] = self.weights[0] - learn * (1 / (len(self.train_data))) * error * result
                for i in range(40):
                    self.weights[i + 1] = self.weights[i + 1] - learn * (1 / (len(self.train_data))) * error * row[i]
        return self.weights

    def predict(self, row):
        # Store the initial weight
        weight = self.weights[0]
        for each in range(40):
            # Using the logistic function, calculate the predicted output
            weight += (self.weights[each + 1] * row[each])
        if weight < 0:
            return 1.0 - 1 / (1.0 + exp(weight))
        else:
            return 1.0 / (1.0 + exp(-weight))

    def main(self):
        learningRate = 0.2
        iterations = 10
        # Returns the list of weights
        self.weights = self.gradient_descent(learningRate, iterations)
        predictions = list()
        for row in range(len(self.test_data)):
            print("Predicting {:3.2%}".format(row / (len(self.test_data))), end="\r")
            row = self.test_data.iloc[row]
            prediction = self.predict(row, self.weights)
            # Rounds prediction result to 2 decimal places.
            if prediction > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        self.log_reg_accuracy(predictions, self.test_data)
        self.save_instance()


