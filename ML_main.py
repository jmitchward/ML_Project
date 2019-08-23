# Machine Learning Main
# The control base for the machine learning program. Class constructs a menu and is
# responsible for the flow of program itself.
# Inherits ML_DFM
# Calls ML_NB, ML_DT, ML_LR

import pandas as pd
import ML_LR
import ML_NB
import ML_DT
import ML_DFM

print("Welcome. The default dataset is loaded. ")


# It makes more sense for the menu to inherit the machine learning class than anything else

class menu(ML_DFM.df_manage):

    def __init__(self):
        # Standard Lists
        self.train_class = []
        self.test_class = []
        self.classifiers = []
        # Used in run_predictions
        self.predict_this = []
        # Used in ML_DFM.feature_define() and ML_DFM.store_values()
        self.feature_values = {}
        # Used in ML_DFM.feature.define() and ML_DFM.get_single()
        self.feature_names = []
        # Used in format_chain() and format_dataset()
        self.skip_check = "no"
        # Used in format_chain()
        self.features = []

        self.train_data = pd.read_csv('./ml_data/census_income_real.data', header=None)
        # http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz
        self.test_data = pd.read_csv('./ml_data/census_income_test.test', header=None)
        # https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz

        # self.train_data = self.train_data[:1000]
        # self.test_data = self.test_data[:1000]

        # Algorithms run on self.data. Begin using the training set.
        self.data = self.train_data
        self.menu()

    def format_chain(self):
        # Need to skip the test dataset discovery as it will need to be the same as the training dataset
        # Why not access the entire dataframe feature by feature then run through that. May be faster
        # Sort the dataset features into categorical and numerical
        if self.skip_check == "no":
            self.find_features()
        # Encode binary classifier into 0 or 1
        self.encode_values(self.features[2])
        # Separate classifier from dataset
        self.classifiers = self.data.iloc[:][self.features[2]]
        # Drop classifier from dataset
        self.data = self.data.drop([self.features[2]], axis=1)
        # Standardize dataset
        self.format_data()

    def import_dataset(self):
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

    def format_dataset(self):
        choice_format = input("Which dataset would you like to format?")

        if choice_format.lower() == "train data":
            print("Formatting train data")
            self.data = self.train_data
            self.format_chain()
            self.skip_check = "yes"
            self.train_data = self.data
            self.train_class = self.classifiers

        elif choice_format.lower() == "test data":
            print("Formatting test data...")
            self.data = self.test_data
            self.format_chain()
            self.test_data = self.data
            self.test_class = self.classifiers

        elif choice_format.lower() == "both":
            print("Formatting both...")
            print("Training Data:")
            self.data = self.train_data
            self.format_chain()
            self.skip_check = "yes"
            self.train_data = self.data
            self.train_class = self.classifiers

            print("Test Data:")
            print("Formatting test data...")
            self.data = self.test_data
            self.format_chain()
            self.test_data = self.data
            self.test_class = self.classifiers

        self.menu()

    # Run the naming of features function
    def run_ml_fn(self):
        self.feature_define()

    # Run the logistic regression function
    def run_ml_lr(self):
        self.data = self.train_data
        self.standardize_data()
        self.data = self.test_data
        self.standardize_data()

        print("Beginning Logistic Regression.")
        self.log_reg = ML_LR.logistic_regression(self.train_data, self.test_data, self.train_class, self.test_class)
        self.save_instance()
        self.menu()

    # Run the decision tree function
    def run_ml_dt(self):
        print("Beginning Decision Tree.")
        self.d_tree = ML_DT.decision_tree.main(self.train_data, self.test_data, self.train_class, self.test_class)
        self.menu()

    # Run the naive bayes function
    def run_ml_nb(self):
        print("Beginning Naive Bayes.")
        self.n_bayes = ML_NB.naive_bayes(self.train_data, self.test_data, self.train_class, self.test_class)
        self.menu()

    def run_predictions(self, algo_choice):
        if not self.feature_names:
            print("Please label columns to improve readability.")
            self.run_ml_fn()
        choice_predict = input("Individual or group prediction?")
        if choice_predict == "individual":
            self.predict_this = self.get_single()
            self.skip_check = "yes"
            self.format_chain(self.predict_this)

            # Dont get overwhelmed. For log reg run the predictor by itself
            # For Naive Bayes run the predictor by itself
            # Need to make sure formatting is correct. Maybe a dictionary of changed values.
            # Need to make sure its categorically formatted the same way as the rest
        elif choice_predict == "dataset":
            self.import_dataset()

    # Iterate over the dataset features asking for the values
    # of this singular entry
    # THEN
    #            log_reg.predict()
    # This is calling something that was not defined with this is chosen first
    # While respecting the parameter of the function call
    #        elif choice_predict == "group":
    # Request either a pre defined new list of entries or manual input

    # In order to predict for any new input, there needs to be established
    # a dictionary correlating the encoded values to the actual categorical
    # feature.

    # That is best done while encoding is happening

    def menu(self):
        print("1. Import Data")
        print("2. Format Data")
        print("3. Name Features")
        print("4. Run Logistic Regression")
        print("5. Run Naive Bayes")
        print("6. Run Decision Tree")
        print("7. Run Predictions ")
        print("8. Exit")

        next_choice = input("What would you like to do?")

        self.menu_select(next_choice)

    def menu_select(self, choice):
        if choice.lower == "import data" or str(choice) == "1":
            self.import_dataset()
        elif choice.lower() == "format data" or str(choice) == "2":
            self.format_dataset()
        elif choice.lower() == "name features" or str(choice) == "3":
            self.run_ml_fn()
        elif choice.lower() == "run logistic regression" or str(choice) == "4":
            self.run_ml_lr()
        elif choice.lower() == "run naive bayes" or str(choice) == "5":
            self.run_ml_nb()
        elif choice.lower() == "run decision tree" or str(choice) == "6":
            self.run_ml_dt()
        # elif choice.lower() == "run predictions" or str(choice) == "7":
        #    self.run_predictions(choice)
        elif choice.lower() == "exit" or str(choice) == "8":
            exit()
        else:
            print("Invalid selection.")
            self.menu()


menu()
