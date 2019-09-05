# Machine Learning-  Dataframe Sort
# Receives the dataset and then parses it splitting the categorical and numerical
# features into two separate lists.
# Called by ML_DFM


def df_discovery(data):
    # List for each of the feature types
    categorical = []
    # Number of features in the dataset

    data_type = input("Will this be supervised?")
    # Ideally a switch for unsupervised which does not feature a classifier
    if data_type.lower() == "yes":
        classifier = input("What column will the classifier be found?")
        classifier = int(classifier)

    #    elif data_type.lower() == "unsupervised":
    #        classifier = 999
    #    else:
    #        print("Invalid selection.")
    #        return

    data_search = int(len(data) / 25)
    print("Beginning discovery...")
    for every in range(len(data.columns)):
        for each in range(data_search):
            # for each column, use every row up to a 25th of the dataset
            if type(data.iloc[each][every]) == str:
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
    for feature in range(len(categorical)):
        print(categorical[feature], end=" ")

    print("\nDiscovered", len(numerical), " numerical features.")
    for features in range(len(numerical)):
        print(numerical[features], end=" ")

    #    doubleCheck = input("Is this correct?")

    #    if doubleCheck.lower() == "yes":
    return categorical, numerical, classifier

#    else:
#       If the features have not been properly defined, extend the search parameters
#        data_search = int(len(data) / 10)
#        dataset_discovery(data)

