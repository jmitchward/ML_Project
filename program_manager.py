# Machine Learning Main
# The control base for the machine learning program. Class constructs a menu and is
# responsible for the flow of program itself.

import pandas as pd
import logistic_regression
import naive_bayes
import decision_tree
import database_manager
import prediction_manager
import create_database
import logging


class pg_manage(database_manager.db_manage):

    def __init__(self):

        logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s)')

        self.ml_instance = './covid_data/ml_instance'
        self.nb_path = './covid_data/nb_instance'
        self.lr_path = './covid_data/lr_instance'
        self.dt_path = './covid_data/dt_instance'
        
        super(pg_manage, self).__init__()

        self.menu()

    def manage_dataset(self):
        print('''Datasets Available:
-Train Data
-Test Data
-Both
-Return ''')
        to_format = input("Which dataset would you like to format?")

        if to_format.lower() == "train data":
            print("Formatting train data")
            self.data = self.train_data
            self.format_chain()
            self.skip_check = "yes"
            self.train_data = self.data
            self.train_class = self.classifiers

        elif to_format.lower() == "test data":
            print("Formatting test data...")
            self.data = self.test_data
            self.format_chain()
            self.test_data = self.data
            self.test_class = self.classifiers

        elif to_format.lower() == "both":
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

        elif to_format.lower() == "return":
            self.menu()

        else:
            print("Invalid selection.")
            self.manage_dataset()

        self.save_instance(self, self.ml_instance)
        self.menu()

    def run_ml_fn(self):
        self.name_data()
        self.save_instance(self, self.ml_instance)
        self.menu()

    def run_ml_lr(self):
        print("Beginning Logistic Regression.")
        self.standardize_data()
        self.data = self.train_data
        self.standardize_data()
        lr_instance = logistic_regression.logistic_regression(self.train_data, self.test_data, self.train_class,
                                                              self.test_class)
        self.save_instance(lr_instance, self.lr_path)
        self.menu()

    def run_ml_dt(self):
        self.format_chain()
        print("Beginning Decision Tree.")
        dt_instance = decision_tree.decision_tree.main(self.train_data, self.test_data, self.train_class,
                                                       self.test_class)
        self.save_instance(dt_instance, self.dt_path)
        self.menu()

    def run_ml_nb(self):
        self.normalize_data()
        self.data = self.train_data
        self.normalize_data()
        print("Beginning Naive Bayes.")
        nb_instance = naive_bayes.naive_bayes(self.train_data, self.test_data, self.train_class, self.test_class)
        self.save_instance(nb_instance, self.nb_path)
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
        print('''1. Import Data
2. Format Data
3. Prune Data
4. Name Features
5. Run Logistic Regression
6. Run Naive Bayes
7. Run Decision Tree
8. Run Predictions 
9. Load Previous State
10. Exit ''')

        next_choice = input("What would you like to do?")

        self.menu_select(next_choice)

    def menu_select(self, choice):
        if choice.lower == "import data" or str(choice) == "1":
            self.get_dataset()
        elif choice.lower() == "format data" or str(choice) == "2":
            self.manage_dataset()
        elif choice.lower() == "prune data" or str(choice) == "3":
            self.prune_data()
            self.menu()
        elif choice.lower() == "name features" or str(choice) == "4":
            self.run_ml_fn()
        elif choice.lower() == "run logistic regression" or str(choice) == "5":
            self.run_ml_lr()
        elif choice.lower() == "run naive bayes" or str(choice) == "6":
            self.run_ml_nb()
        elif choice.lower() == "run decision tree" or str(choice) == "7":
            self.run_ml_dt()
        elif choice.lower() == "run predictions" or str(choice) == "8":
            self.run_predictions()
        elif choice.lower() == "load state" or str(choice) == "9":
            self.load_state = self.load_instance(self.ml_instance)
            self.load_state.menu()
        elif choice.lower() == "exit" or str(choice) == "10":
            exit()
        else:
            print("Invalid selection.")
            self.menu()


if __name__ == '__main__':
    pg_manage()
