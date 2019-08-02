import numpy as np
import pandas as pd
from math import sqrt
from math import pi
from math import exp
from random import randrange

# using Pandas, retrieve the data sets from the UCI Repository.,
print('Data retrieved.')
trainData = pd.read_csv('./ml_data/census_income_real.data', header=None)
testData = pd.read_csv('./ml_data/census_income_test.test', header=None)


# trainData = pd.read_csv(
#    'http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz', header=None)
# testData = pd.read_csv(
#    'https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz', header=None)


# Encodes values by converting them to category type and replacing the value with its category index.,
def encodeValue(data, column):
    data[column] = data[column].astype('category')
    data[column] = data[column].cat.codes
    data[column] = data[column].astype('int')


def formatData(dataset):
    # 41 total features of the data set, 31 require encoding,
    numFeatures = [0, 2, 3, 5, 16, 17, 18, 24, 30, 39]
    catFeatures = [1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 31, 32, 33,
                   34, 35, 36, 37, 38, 40]
    questionFeatures = [25, 26, 27, 29]
    print(' Encoding categorical features.')
    for each in catFeatures:
        encodeValue(dataset, each)
        # Encode class value.,
    encodeValue(dataset, 41)
    for value in questionFeatures:
        dataset[value].replace(' ?', dataset.describe(include='all')[value][2], inplace=True)
    # Take the categorical features, feed them to the encode function.,


#    print(' Standardizing numerical features.'),
#    for each in numFeatures:,
#        mean, std = dataset[each].mean(), dataset[each].std(),
#        dataset.iloc[:, each] = (dataset[each] - mean)/std,
#    #Standardize categorical features,
#    print(' Standardizing categorical features.'),
#    for each in catFeatures:,
#        dataset.loc[:,each] = (dataset[each] - dataset[each].mean())/dataset[each].std()  ,

# Calculates probability,
def calcProb(value, mean, sdev):
    secondHalf = exp(-((value - mean) ** 2 / (2 * sdev ** 2)))
    return 1 / (sqrt(2 * pi) * (sdev ** 2)) * secondHalf


# Calculates the mean and standard deviation for each column,
def easyCalc(dataset):
    summaries = list()
    for each in range(41):
        mean = dataset[each].mean()
        sdev = dataset[each].std()
        # creates a list containing the mean and s-dev for each feature in the set.,
        summaries.append([mean, sdev])
    return summaries


# Calculates accuracy by comparing number of correctly predicted against their actual values.,
def accuracy(predict, data):
    print('Beginning accuracy rating.')
    correct = 0
    dSize = len(data)
    for i in range(dSize):
        print("Progress {:3.2%}".format(i / dSize), end="\r")
        if data.iloc[i][41] == predict[i]:
            correct += 1
    print('')
    print('Accuracy:', round(((correct / dSize) * 100.0)), '%')


# Calculates the probability that a given sample belongs to a class ,
def NBcalcClassProb(row, aboveFiveProb, belowFiveProb, summaries):
    # Establish initial probability,
    aboveProb = aboveFiveProb
    belowProb = belowFiveProb
    # for each feature, calculating its probability of belonging to each class,
    for eachValue in range(41):
        aboveProb *= (calcProb(row[eachValue], summaries[eachValue][0], summaries[eachValue][1]))
        belowProb *= (calcProb(row[eachValue], summaries[eachValue][0], summaries[eachValue][1]))
    return aboveProb, belowProb


# Determines which class a given sample belongs to,
def NBpredict(row, summaries):
    # Returns the probability of each classification
    classProb = (encodedTrain[41].value_counts() / len(encodedTrain))
    # Probability of a sample belonging to 50000+,
    aboveFiveProb = classProb[0]
    # Probability of a sample belonging to -50000,
    belowFiveProb = classProb[1]
    probabilities = (NBcalcClassProb(row, aboveFiveProb, belowFiveProb, summaries))
    if probabilities[0] > probabilities[1]:
        return 0
    else:
        return 1


def naiveBayes(data, testData):
    predictions = list()
    dataSize = len(testData)
    summaries = easyCalc(data)
    for i in range(dataSize):
        print("Progress {:3.2%}".format(i / (len(testData) * 2)), end="\r")
        row = testData.iloc[i]
        output = NBpredict(row, summaries)
        predictions.append(output)
    accuracy(predictions, testData)


encodedTrain = trainData
encodedTest = testData
print('Processing traininng data.')
formatData(encodedTrain)
print('Processing test data.')
formatData(encodedTest)
# trainSet = trainData[:10000],
# testSet = testData[:20000],
print('Beginning Naive Bayes algorithm.')
naiveBayes(encodedTrain, encodedTest)
