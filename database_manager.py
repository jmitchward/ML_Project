# Machinen Learning - Dataframe Management
# A class defining the functions used to manipulate the dataset.
# Inherited by ML_main

import database_setup
import textwrap


class df_manage:

    def find_features(self):
        # Limits access to the DFS file
        self.features = database_setup.df_discovery(self.data)
        self.store_values()

    def encode_values(self, column):
        # Cast the data frame as category
        self.data[column] = self.data[column].astype('category')
        # Change every value in its respective categorical value
        self.data[column] = self.data[column].cat.codes
        # Cast the new values as int
        self.data[column] = self.data[column].astype('int')

    def format_data(self):
        # Categorical features = self.features[0]
        # Numerical Features = self.features[1]

        print("\nSearching for illegal characters.")
        # for value in self.features[0]:
        #   temp_data = self.data[:].astype('category')
        #   self.data[value].replace(' ?', temp_data.describe(include='all')[value][2], inplace=True)
        print('Encoding categorical features.')
        for each in self.features[0]:
            self.encode_values(each)
        return self.data

    def standardize_data(self):
        # Categorical features = self.features[0]
        # Numerical Features = self.features[1]

        print('Standardizing numerical features.')
        for each in self.features[1]:
            # Calculate the mean and standard deviation
            mean, std = self.data[each].mean(), self.data[each].std()
            # Use these values to standardize numerical features
            self.data.iloc[:, each] = (self.data[each] - mean) / std

        print('Standardizing categorical features.')
        for each in self.features[0]:
            # Use these values to standardize categorical features
            self.data.loc[:, each] = (self.data[each] - self.data[each].mean()) / self.data[each].std()

    def feature_define(self):
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

    def store_values(self):
        temp_data = self.data[:].astype('category')
        # Store column values in a list of lists
        for catFeatures in self.features[0]:
            current_feature = list(temp_data[catFeatures].cat.categories)
            # Create dictionary of categorical features and their list of values
            self.feature_values.update({catFeatures: current_feature})

    def get_single(self):
        for each in range(len(self.feature_names)):
            # For each feature, which there are 41 in the default
            # Print the possible stored values for user input to be selected
            # from.
            current_name = self.feature_names[each]
            print("Options for", current_name)
            if each in self.feature_values:
                for every in range(len(self.feature_values[each])):
                    print(every, textwrap.fill(self.feature_values[each][every], 40))
                GTP = input("Select or enter a value:")
                # self.predict_this.append(self.feature_values[each][GTP])
                # self.single_encode(each)
            else:
                GTP = input("Select or enter a value:")
            GTP = int(GTP)
            self.predict_this.append(GTP)
        self.data = self.predict_this

    def single_encode(self, each_feature):
        # feature_values : Dictionary list of values group by features
        # Established by the store_values() function
        # single: An entry of data to be processed
        # Cotains each possile value of the given feature indices
        # I.E. {2: 'M, F, U'}
        # faetures_names: List of the feature titles
        for each_index in range(len(self.feature_values[each_feature])):
            mutate_value = self.feature_values[each_feature][each_index].strip(' ')
            mutate_value = self.feature_values[each_feature][each_index].lower()
            if self.predict_this[each_feature] == mutate_value:
                self.predict_this[each_feature] = each_index
