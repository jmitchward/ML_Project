# Machine Learning - Dataframe Management
# A class defining the functions used to manipulate the dataset.
# Inherited by ML_main

import database_setup
import logging
import numpy as np
import pandas as pd


class db_manage(database_setup.create_db):
    logging.basicConfig(level=logging.DEBUG, format='\n %(asctime)s - %(levelname)s - %(message)s')

    def format_chain(self, data):
        if self.skip_check == "no":
            logging.debug('Skip_check is ' + str(self.skip_check))
            self.learning_method(data)
            # Prompt user for learning method, supervised or unsupervised
            # This method returns a list of features, separated based on numerical or categorical
            data, classifiers = self.format_data(data)
            # Do the steps to format the data
            # This method returns the database and the classifiers, which were pulled out of the database
            return data, classifiers
        else:
            data, classifiers = self.format_data(data)
            return data, classifiers

    def format_data(self, data):
        # Categorical features = self.features[0]
        # Numerical Features = self.features[1]

        # print("\nSearching for illegal characters.")
        # for value in self.features[0]:
        #     temp_data = data[:].astype('category')
        #     data[value].replace(' ?', temp_data.describe(include='all')[value][2], inplace=True)

        print('Encoding categorical features...')
        logging.debug('Categorical Features: ' + str(len(self.features[0])))
        data = self.encode_data(data, self.features[0])
        # Encode binary classifier into 0 or 1

        logging.debug('Classifier: ' + str(self.features[2]))
        data = self.encode_data(data, self.features[2])
        # Separate classifier from dataset

        logging.debug('Classifier being stored in self.classifiers')
        classifiers = data.iloc[:][self.features[2]]
        # Drop classifier from dataset

        logging.debug('Classifier being dropped from self.data')
        data = data.drop([self.features[2]], axis=1)

        return data, classifiers

    def standardize_data(self, data):
        print('Standardizing data.')
        for each in (self.features[0]):
            data.iloc[:, self.features[0].index(each)] = (data.iloc[:, self.features[0].index(each)] - data.iloc[self.features[0].index(each)].mean()) / data.iloc[self.features[0].index(each)].std()
        for each in self.features[1]:
            data.iloc[:, self.features[1].index(each)] = (data.iloc[:, self.features[1].index(each)] - data.iloc[self.features[1].index(each)].mean()) / data.iloc[self.features[1].index(each)].std()
        return data

    @staticmethod
    def encode_data(data, features):
        logging.debug(type(features))
        if isinstance(features, np.int64) or isinstance(features, np.str):
            logging.debug('Encoding the classifier ' + str(features))
            # Cast the data frame as category
            data[features] = data[features].astype('category')
            # Change every value in its respective categorical value
            data[features] = data[features].cat.codes
            # Cast the new values as int
            data[features] = data[features].astype('int')
            logging.debug('Encode_data exited')
            return data
        else:
            logging.debug('Encoding categorical features')
            for each in features:
                # Cast the data frame as category
                logging.debug('Working on Feature ' + str(each))
                data[each] = data[each].astype('category')
                # Change every value in its respective categorical value
                data[each] = data[each].cat.codes
                # Cast the new values as int
                data[each] = data[each].astype('int')
            logging.debug('Encode_data exited')
            return data

    @staticmethod
    def normalize_data(data):
        # XNEW = (VALUE - VALUE(MIN)) / (VALUE(MAX))-VALUE(MIN))
        logging.debug('Normalize_data entered.')
        list_of_maxs = data.max()
        list_of_mins = data.min()
        for every_column in range(len(data.columns)):
            data.iloc[:, every_column] = (data.iloc[:, every_column] - list_of_mins[every_column]) / (
                    list_of_maxs[every_column] - list_of_mins[every_column])
        return data

    @staticmethod
    def prune_data(data):
        indices = 0
        while indices < len(data.columns):
            print("Feature:", data.columns[indices])
            print(data.iloc[:5][data.columns[indices]])
            check = input("Feature " + str(indices) + ". Remove this column? ")
            if check.lower() == 'yes':
                print("Removing", data.columns[indices])
                data = data.drop(data.columns[indices], axis=1)
                data.reset_index(drop=True)
                indices += 1
            else:
                indices += 1

        return data

    def replace_column_names(self, data):
        i = 0
        print("Replacing column names with indexes.")
        for each_column in data.columns:
            self.column_names.append(each_column)
            data.rename(columns={each_column: i}, inplace=True)
            i += 1
        data.set_index
        return data
