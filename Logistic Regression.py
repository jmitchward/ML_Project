# Jason Ward 2017-2019

from math import exp

import basic_func


class logistic_regression(basic_func):
    # Returns list of 41 zeros
    weights = [0.0] * 41

    def gradient_descent(self, data, learn, iterations):

        # For the pre-determined number of iterations
        for eachIter in range(iterations):
            sumError = 0
            for row in range(len(data)):
                print("Updating Weights {:3.2%}".format(row / (len(data))), end="\r")
                row = data.iloc[row]
                result = data.predict(row, self.weights)
                error = (result - row[41])
                sumError += (.5) * (error ** 2)
                self.weights[0] = self.weights[0] - learn * (1 / (len(data))) * error * result
                for i in range(40):
                    self.weights[i + 1] = self.weights[i + 1] - learn * (1 / (len(data))) * error * row[i]
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

    def main(self, data, testData):
        learningRate = 0.2
        iterations = 10
        # Returns the list of weights
        self.weights = data.gradient_descent(data, learningRate, iterations)
        predictions = list()
        for row in range(len(testData)):
            print("Predicting {:3.2%}".format(row / (len(testData))), end="\r")
            row = testData.iloc[row]
            prediction = data.predict(row, self.weights)
            # Rounds prediction result to 2 decimal places.
            if prediction > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        basic_func.log_reg_accuracy(predictions, testData)


print('Beginning Logistic Regression algorithm.')
log_reg = logistic_regression()
log_reg.main(basic_func.encoded_training_data, basic_func.encoded_test_data)
