# Machine Learning - Dataframe Management
# A class defining the functions used to manipulate the dataset.
# Inherited by ML_main

import create_database
import logging


class db_manage(create_database.create_db):

    logging.basicConfig(level=logging.DEBUG, format='\n %(asctime)s - %(levelname)s - %(message)s')

    def setup_dataset(self):
        self.learning_method(self.data)

    def format_chain(self):
        if self.skip_check == "no":
            logging.debug('Skip_check is ' + str(self.skip_check))
            self.setup_dataset()
            logging.debug('Format chain entered on train_data')
            self.format_data()
            self.data = self.test_data
            logging.debug('Format chain entered on test_data')
            self.format_data()
            # self.standardize_data()
            logging.debug('Data reset to train_data')
            self.data = self.train_data
            # self.standardize_data()
        else:
            self.format_data()

    def encode_data(self, features):
        if type(features) is int:
            logging.debug('Encoding the classifier ' + str(features))
            # Cast the data frame as category
            self.data[features] = self.data[features].astype('category')
            # Change every value in its respective categorical value
            self.data[features] = self.data[features].cat.codes
            # Cast the new values as int
            self.data[features] = self.data[features].astype('int')
        else:
            logging.debug('Encoding categorical features')
            for each in features:
                # Cast the data frame as category
                self.data[each] = self.data[each].astype('category')
                # Change every value in its respective categorical value
                self.data[each] = self.data[each].cat.codes
                # Cast the new values as int
                self.data[each] = self.data[each].astype('int')
        logging.debug('Encode_data exited')

    def format_data(self):
        # Categorical features = self.features[0]
        # Numerical Features = self.features[1]

        # print("\nSearching for illegal characters.")
        # for value in self.features[0]:
        #    temp_data = self.data[:].astype('category')
        #    self.data[value].replace(' ?', temp_data.describe(include='all')[value][2], inplace=True)
        print('Encoding categorical features...')
        logging.debug('Categorical Features: ' + str(len(self.features[0])))
        self.encode_data(self.features[0])
        # Encode binary classifier into 0 or 1
        logging.debug('Classifier: ' + str(self.features[2]))
        self.encode_data(self.features[2])
        # Separate classifier from dataset
        logging.debug('Classifier being stored in self.classifiers')
        self.classifiers = self.data.iloc[:][self.features[2]]
        # Drop classifier from dataset
        logging.debug('Classifier being dropped from self.data')
        self.data = self.data.drop([self.features[2]], axis=1)
        # Normalize dataset
        # logging.debug('Normalize_data being called from format_data')
        # self.normalize_data()
        # OR
        # Standardize dataset
        # self.standardize_data()

    def normalize_data(self):
        # XNEW = (VALUE - VALUE(MIN)) / (VALUE(MAX))-VALUE(MIN))
        logging.debug('Normalize_data entered.')
        list_of_maxs = self.data.max()
        logging.debug('Maximums ' + str(list_of_maxs))
        list_of_mins = self.data.min()
        logging.debug('Minimums ' + str(list_of_mins))
        for every_column in self.data.columns:
            self.data.iloc[:, every_column] = (self.data.iloc[:, every_column] - list_of_mins[every_column]) / (
                        list_of_maxs[every_column] - list_of_mins[every_column])

    def standardize_data(self):
        print('Standardizing data.')
        for each in (self.features[0] + self.features[1]):
            self.data.iloc[:, each] = (self.data.iloc[:, each] - self.data[each].mean()) / self.data[each].std()

    def prune_data(self):
        indices = 0
        while indices < len(self.data.columns):
            print("Feature:", self.data.columns[indices])
            print(self.data.iloc[:5][self.data.columns[indices]])
            check = input("Remove this column? ")
            if check.lower() == 'yes':
                print("Removing", self.data.columns[indices])
                self.data = self.data.drop(self.data.columns[indices], axis=1)
                self.data.reset_index(drop=True)
                indices += 1
            else:
                indices += 1

        return self.data

    def backup_database(self):
        temp_data = self.data[:].astype('category')
        # Store column values in a list of lists
        for catFeatures in self.features[0]:
            current_feature = list(temp_data[catFeatures].cat.categories)
            # Create dictionary of categorical features and their list of values
            self.feature_values.update({catFeatures: current_feature})
