import pickle
import pandas as pd


class create_db:

    def __init__(self):
        self.skip_check = "no"
        # Lists
        self.train_class = []
        self.test_class = []
        self.classifiers = []

        self.feature_names = []
        self.features = []
        # Dictionaries
        self.feature_values = {}
        # Misc Values
        # self.train_data = pd.read_csv('./titanic/train.csv')
        # self.train_data = pd.read_pickle('./covid_data/new_dataset.pkl')
        self.train_data = pd.read_csv('./ml_data/census_income_real.data', header=None)
        # http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz
        # self.test_data = pd.read_csv('./titanic/test.csv')
        # self.test_data = pd.read_pickle('./covid_data/new_dataset.pkl')
        self.test_data = pd.read_csv('./ml_data/census_income_test.test', header=None)
        # https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz
        self.data = self.train_data
        # START

    def learning_method(self, data):
        data_type = input("Will this be supervised? Yes/No: ")
        # Ideally a switch for unsupervised which does not feature a classifier
        if data_type.lower() == "yes":
            classifier = input("What column will the classifier be found?")
            classifier = int(classifier)
            self.features = self.supervised_learning(data, classifier)
        else:
            print("Invalid selection.")
            self.learning_method(data)

    def supervised_learning(self, data, classifier):
        categorical = []
        # Number of features in the dataset
        data_search = int(len(self.data))
        print("Beginning discovery...")
        for every in range(len(self.data.columns)):
            for each in range(self.data_search):
                # for each column, use every row up to a 25th of the dataset
                if type(self.data.iloc[each][every]) == str:
                    # If any value within that column is a string, it categorical
                    categorical.append(every)
                    # Add it to the list then break to the next column
                    break
                    # If it is a not a string, then it is a number

        # Make a list of the remaining, non-categorical features
        numerical = list(set(data.columns) - set(categorical))

        # Check if the classifier has been placed in either of the created lists
        # If it has been, remove it
        for eachFeature in categorical:
            if eachFeature == classifier:
                categorical.remove(classifier)
        for everyFeature in numerical:
            if everyFeature == classifier:
                numerical.remove(classifier)

        print("Discovered", len(categorical), "categorical features.")
        #    for feature in range(len(categorical)):
        #        print(categorical[feature], end=" ")

        print("\nDiscovered", len(numerical), "numerical features.")
        #    for features in range(len(numerical)):
        #        print(numerical[features], end=" ")

        #    doubleCheck = input("Is this correct?")

        #    if doubleCheck.lower() == "yes":
        return categorical, numerical, classifier

    def get_dataset(self):
        TTD = input("Would you like to import the training and test data separately or split automatically?")

        if TTD == "separately":
            url = input("Enter the URL for the training data:")
            # Import dataset. Input cleaning needed
            self.train_data = pd.read_csv(url)
            self.data = self.train_data
            # Run algorithms needed to format
            self.format_chain()
            # Skip the future test set, which will have the same features
            self.skip_check = "yes"
            # Ensure train_data is properly updated before switching to test data next
            self.train_data = self.data
            # Ensure training class values are updated before switching to test class
            self.train_class = self.classifiers

            url = input("Enter the URL for the test data:")
            self.test_data = pd.read_csv(url)
            self.data = self.test_data
            self.format_chain()
            self.test_data = self.data
            self.test_class = self.classifiers

        self.menu()

    def name_data(self):
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

    def load_instance(self, file_path):
        with open(file_path, 'rb') as load_file:
            saved_dataset = pickle.load(load_file)
            return saved_dataset

    def save_instance(self, the_object, file_path):
        # Stores the instance for a multiple classification structure
        with open(file_path, 'wb') as the_file:
            pickle.dump(the_object, the_file)
