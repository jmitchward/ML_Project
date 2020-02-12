# Machine Learning Base
# The back bone functions shared by each of the machine learning algorithms
# Called by ML_NB, ML_LR and ML_DT

from math import sqrt
from math import pi
from math import exp


class machine_learning:
    numFeatures = []
    catFeatures = []
    questionFeatures = []



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
        for each in range(len(data.columns)):
            mean = data[each].mean()
            sdev = data[each].std()
            # creates a list containing the mean and s-dev for each feature in the set.
            summaries.append([mean, sdev])
        return summaries

    @staticmethod
    def accuracy(classifier, predict):
        print('Beginning accuracy rating.')
        correct = 0
        for i in range(len(classifier)):
            # Iterate over the classifier list for the dataset
            # If it matches the value in the predicted list, rate it
            print("Scoring {:3.2%}".format(i / (len(classifier))), end="\r")
            if classifier[i] == predict[i]:
                correct += 1
        print('Accuracy:', round(((correct / len(classifier)) * 100.0)), '%')
        return round(((correct / len(classifier)) * 100.0))
