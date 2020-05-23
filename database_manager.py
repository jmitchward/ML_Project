# Machinen Learning - Dataframe Management
# A class defining the functions used to manipulate the dataset.
# Inherited by ML_main

import create_database
import pandas as pd


class db_manage(create_database.create_db):

    def setup_dataset(self):
        create_database.learning_method(self.data)

    def format_chain(self):
        if self.skip_check == "no":
            self.setup_dataset()
            self.data = self.test_data
            self.standardize_data()
            self.data = self.train_data
            self.standardize_data()
        self.format_data()

    def encode_data(self, column):
        if type(column) is int:
            # Cast the data frame as category
            self.data[column] = self.data[column].astype('category')
            # Change every value in its respective categorical value
            self.data[column] = self.data[column].cat.codes
            # Cast the new values as int
            self.data[column] = self.data[column].astype('int')
        else:
            for each in column:
                # Cast the data frame as category
                self.data[each] = self.data[each].astype('category')
                # Change every value in its respective categorical value
                self.data[each] = self.data[each].cat.codes
                # Cast the new values as int
                self.data[each] = self.data[each].astype('int')

    def format_data(self):
        # Categorical features = self.features[0]
        # Numerical Features = self.features[1]

        # print("\nSearching for illegal characters.")
        # for value in self.features[0]:
        #    temp_data = self.data[:].astype('category')
        #    self.data[value].replace(' ?', temp_data.describe(include='all')[value][2], inplace=True)
        print('Encoding categorical features...')
        self.encode_data(self.features[0])
        # Encode binary classifier into 0 or 1
        self.encode_data(self.features[2])
        # Separate classifier from dataset
        self.classifiers = self.data.iloc[:][self.features[2]]
        # Drop classifier from dataset
        self.data = self.data.drop([self.features[2]], axis=1)
        # Normalize dataset
        self.normalize_data()
        # OR
        # Standardize dataset
        # self.standardize_data()
        return self.data

    def normalize_data(self):
        # XNEW = (VALUE - VALUE(MIN)) / (VALUE(MAX))-VALUE(MIN))
        list_of_maxs = self.data.max()
        list_of_mins = self.data.min()
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
