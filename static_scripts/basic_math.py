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

    # Calculates the mean and standard deviation for each column
    @staticmethod
    def basic_calc(data):
        summaries = list()
        for each in range(len(data.columns)):
            mean = data.iloc[each].mean()
            sdev = data.iloc[each].std()
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
        print('Accuracy:', round(((correct / len(classifier)) * 100.0)),'%')
        return round(((correct / len(classifier)) * 100.0))
