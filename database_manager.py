# Machinen Learning - Dataframe Management
# A class defining the functions used to manipulate the dataset.
# Inherited by ML_main

import database_setup


class df_manage:

    def __init__(self):
        self.feature_values = {}

    def call_setup(self):
        self.features = database_setup.df_discovery(self.data)
    #        self.backup_database()

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
        max_array = self.data.max()
        min_array = self.data.min()
        for every_column in self.data.columns:
            self.data.iloc[:, every_column] = (self.data.iloc[:, every_column] - min_array[every_column]) / (max_array[every_column] - min_array[every_column])

    def standardize_data(self):
        print('Standardizing data.')
        for each in (self.features[0] + self.features[1]):
            self.data.iloc[:, each] = (self.data.iloc[:, each] - self.data[each].mean()) / self.data[each].std()

    def feature_naming(self):
        # Replace feature names as this function is only called when names are being written.
        self.feature_names.clear()
        for each in range(len(self.data.columns)):
            # feature_values only includes categorical features/columns
            for categories in self.feature_values:
                # Cycle each column that has categorical values
                if each == self.feature_values[categories]:
                    # If the current column is one of those values
                    print(self.feature_values[categories])
                    # Print those values for input clarity
                    column_name = input("Enter the name for this feature:")
                    self.feature_names.append(column_name)
                else:
                    continue
            # Otherwise those columns are numerical
            print(self.data.iloc[:5][each])
            column_name = input("Enter the name for this feature:")
            self.feature_names.append(column_name)

    def backup_database(self):
        temp_data = self.data[:].astype('category')
        # Store column values in a list of lists
        for catFeatures in self.features[0]:
            current_feature = list(temp_data[catFeatures].cat.categories)
            # Create dictionary of categorical features and their list of values
            self.feature_values.update({catFeatures: current_feature})



