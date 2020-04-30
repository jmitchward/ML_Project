# Machine Learning Main
# The control base for the machine learning program. Class constructs a menu and is
# responsible for the flow of program itself.
# Inherits ML_DFM
# Calls ML_NB, ML_DT, ML_LR

import pandas as pd
import logistic_regression
import naive_bayes
import decision_tree
import database_manager
import prediction_manager
import database_setup


class menu(database_manager.df_manage):

    def __init__(self):
        # Lists
        self.train_class = []
        self.test_class = []
        self.classifiers = []

        self.feature_names = []
        self.features = []
        # Dictionaries
        self.feature_values = {}
        # Misc Values
        self.skip_check = "no"

        self.ml_instance = './ml_data/ml_instance'
        self.nb_path = './ml_data/nb_instance'
        self.lr_path = './ml_data/lr_instance'
        self.dt_path = './ml_data/dt_instance'

        #self.train_data = pd.read_csv('divorce.csv', header=None)
        self.train_data = pd.read_csv('./ml_data/census_income_real.data', header=None)
        # http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz
        #self.test_data = pd.read_csv('divorce.csv', header=None)
        self.test_data = pd.read_csv('./ml_data/census_income_test.test', header=None)
        # https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz
        self.data = self.train_data
        # START
        self.menu()

    def format_chain(self):
        if self.skip_check == "no":
            self.call_setup()
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
            print("Formatting training data.")
            self.data = self.train_data
            self.format_chain()
            self.skip_check = "yes"
            self.train_data = self.data
            self.train_class = self.classifiers

            print("Formatting test data.")
            self.data = self.test_data
            self.format_chain()
            self.test_data = self.data
            self.test_class = self.classifiers

        database_setup.save_instance(self, self.ml_instance)
        self.menu()

    def run_ml_fn(self):

        self.feature_naming()
        database_setup.save_instance(self, self.ml_instance)
        self.menu()

    def run_ml_lr(self):
        if self.skip_check == "no":
            self.data = self.test_data
            self.standardize_data()
            self.data = self.train_data
            self.standardize_data()
        print("Beginning Logistic Regression.")
        lr_instance = logistic_regression.logistic_regression(self.train_data, self.test_data, self.train_class,
                                                              self.test_class)
        database_setup.save_instance(lr_instance, self.lr_path)
        self.menu()

    def run_ml_dt(self):
        if self.skip_check == "no":
            self.data = self.test_data
            self.standardize_data()
            self.data = self.train_data
            self.standardize_data()
        print("Beginning Decision Tree.")
        dt_instance = decision_tree.decision_tree.main(self.train_data, self.test_data, self.train_class,
                                                       self.test_class)
        database_setup.save_instance(dt_instance, self.dt_path)
        self.menu()

    def run_ml_nb(self):
        if self.skip_check == "no":
            self.data = self.test_data
            self.standardize_data()
            self.data = self.train_data
            self.standardize_data()
        print("Beginning Naive Bayes.")
        nb_instance = naive_bayes.naive_bayes(self.train_data, self.test_data, self.train_class, self.test_class)
        database_setup.save_instance(nb_instance, self.nb_path)
        self.menu()

    def run_predictions(self):
        predict_type = input("Individual or group prediction?")
        print(" 1. Logistic Regression \n 2. Naive Bayes \n 3. Decision Tree")
        algo = input("Please select an algorithm:")
        results = prediction_manager.predict_manage(predict_type, algo, self.feature_names, self.feature_values)
        if results == 0:
            print("Based on the information given, I predict the person makes less than 50,000 dollars a year.")
        else:
            print("Based on the information given, I predict the person makes more than 50,000 dollars a year.")
        self.menu()

    def menu(self):
        print("1. Import Data")
        print("2. Format Data")
        print("3. Name Features")
        print("4. Run Logistic Regression")
        print("5. Run Naive Bayes")
        print("6. Run Decision Tree")
        print("7. Run Predictions ")
        print("8. Load Previous State")
        print("9. Exit")

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
        elif choice.lower() == "run predictions" or str(choice) == "7":
            self.run_predictions()
        elif choice.lower() == "load state" or str(choice) == "8":
            self.load_state = database_setup.load_instance(self.ml_instance)
            self.load_state.menu()
        elif choice.lower() == "exit" or str(choice) == "9":
            exit()
        else:
            print("Invalid selection.")
            self.menu()


if __name__ == '__main__':
    menu()
