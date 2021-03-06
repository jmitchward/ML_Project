import pandas as pd
import database_setup
import textwrap

import static_scripts.object_persistence


class predict_manage:

    def __init__(self, prediction, algorithm, feature_names, feature_values):

        self.predict_this = []
        self.nb_path = './ml_data/nb_instance'
        self.lr_path = './ml_data/lr_instance'
        self.dt_path = './ml_data/dt_instance'
        self.prediction_method = prediction
        self.prediction_algo = algorithm
        self.feature_names = feature_names
        self.feature_values = feature_values

        self.main()

    def main(self):
        if self.prediction_method == "1":
            # Retrieve a single prediction
            self.single_predictee()
            self.data = pd.DataFrame(self.data)
        # if not self.feature_names:
        # print("Please label columns to improve readability.")
        # self.run_ml_fn()

        if self.prediction_algo.lower == "log reg" or str(self.prediction_algo) == "1":
            saved_dataset = static_scripts.object_persistence.load_instance(self.lr_path)
            for each_predicted in range(len(self.data)):
                saved_dataset.lr_predict(self.data[each_predicted])
            return saved_dataset.predictions
        elif self.prediction_algo.lower == "naive bayes" or str(self.prediction_algo) == "2":
            saved_dataset = static_scripts.object_persistence.load_instance(self.nb_path)
            saved_dataset.nb_predict(self.data)
            return saved_dataset.predictions
        elif self.prediction_algo.lower == "decision tree" or str(self.prediction_algo) == "3":
            saved_dataset = static_scripts.object_persistence.load_instance(self.dt_path)
            saved_dataset.dt_predict(self.data)
            return saved_dataset.predictions
        else:
            self.main()

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
