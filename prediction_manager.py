import pandas as pd
import program_manager
import textwrap


class predict_manage(program_manager.menu):

    def __init__(self):

        nb_path = './ml_data/nb_instance'
        lr_path = './ml_data/lr_instance'
        dt_path = './ml_data/dt_instance'

        choice_predict = input("Individual or group prediction?")
        if choice_predict == "individual":
            # Retrieve a single prediction
            self.single_predictee()
            self.data = pd.DataFrame(self.data)
        # if not self.feature_names:
        # print("Please label columns to improve readability.")
        # self.run_ml_fn()
        print("\nIndividual entry loaded and formatted.\n")
        print("1. Logistic Regression \n 2. Naive Bayes \n 3. Decision Tree \n")
        local_choice = input("Please select an algorithm:")
        if local_choice == "log reg" or int(1):
            if not self.log_reg:
                saved_dataset = self.load_instance(self, lr_path)
                saved_dataset.lr_predictor(self.data)
                # If that fails, kick back to main
            else:
                self.log_reg.predictor(self.data)
        if local_choice == "naive bayes" or int(2):
            saved_dataset = self.load_instance(self, nb_path)
            saved_dataset.nb_predict(self.data)
            print(saved_dataset.predictions)
        if local_choice == "decision tree" or int(3):
            if not self.d_tree:
                saved_dataset = self.load_instance(self, dt_path)
                saved_dataset.dt_predict(self.data)
            else:
                self.saved_instance.predict(self.data)
        elif choice_predict == "dataset":
            self.import_dataset()

    #        elif choice_predict == "group":
    # Request either a pre defined new list of entries or manual input

    # In order to predict for any new input, there needs to be established
    # a dictionary correlating the encoded values to the actual categorical
    # feature.

    # That is best done while encoding is happening

    def single_predictee(self):
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

    def single_encoder(self, each_feature):
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
