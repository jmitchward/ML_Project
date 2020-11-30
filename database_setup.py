import logging
import pandas as pd


class create_db:

    def __init__(self):
        logging.basicConfig(level=logging.DEBUG, format='\n %(asctime)s - %(levelname)s - %(message)s)')

        self.skip_check = "no"
        # Lists
        self.train_class = []
        self.test_class = []
        self.classifiers = []
        self.column_names = []
        self.feature_names = []
        self.features = []
        # Dictionaries
        self.feature_values = {}
        # Misc Values
        self.train_data = pd.read_csv('./ml_data/titanic/train.csv')
        # self.train_data = pd.read_pickle('./covid_data/new_dataset.pkl')
        # self.train_data = pd.read_csv('./ml_data/census_income_real.data', header=None)
        # http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz
        self.test_data = pd.read_csv('./ml_data/titanic/test.csv')
        # self.test_data = pd.read_pickle('./covid_data/new_dataset.pkl')
        # self.test_data = pd.read_csv('./ml_data/census_income_test.test', header=None)
        # https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz
        self.data = self.train_data
        # START

    def learning_method(self, data):
        data_type = input("Will this be supervised? Yes/No: ")
        # Ideally a switch for unsupervised which does not feature a classifier
        if data_type.lower() == "yes":
            classifier = input("What column will the classifier be found?")
            try:
                classifier = data.columns[int(classifier)]
            except ValueError:
                print("Please enter the column number where the classifier can be found.")
                self.learning_method(data)
            logging.debug('Classifier entered: ' + str(classifier))
            self.features = self.supervised_learning(data, classifier)
            return self.features
        else:
            print("Invalid selection.")
            self.learning_method(data)

    def supervised_learning(self, data, classifier):
        categorical = []
        # Number of features in the dataset
        logging.debug("Entering create_db.supervised_learning")
        print("Beginning discovery...")
        for every in range(len(self.data.columns)):
            for each in range(len(self.data)):
                # for each column, use every row up to a 25th of the dataset
                if type(self.data.iloc[each][every]) == str:
                    try:
                        int(self.data.iloc[each][every])
                        break
                    except ValueError:
                        # If any value within that column is a string, it categorical
                        logging.debug('Feature ' + str(every) + ' entered due to ' + str(self.data.iloc[each][every]))
                        categorical.append(data.columns[every])
                        # Add it to the list then break to the next column
                        break
                        # If it is a not a string, then it is a number

        # Make a list of the remaining, non-categorical features
        logging.debug('Categorical' + str(categorical))
        logging.debug('Columns ' + str(data.columns))
        numerical = list(set(data.columns) - set(categorical))
        logging.debug('Numerical' + str(numerical))

        # Check if the classifier has been placed in either of the created lists
        # If it has been, remove it
        for eachFeature in categorical:
            if eachFeature == classifier:
                categorical.remove(classifier)
        for everyFeature in numerical:
            if everyFeature == classifier:
                numerical.remove(classifier)

        logging.debug("Discovered " + str(len(categorical)) + " categorical features.")

        logging.debug("Discovered " + str(len(numerical)) + " numerical features.")

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

    def set_column_names(self):
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
