import pickle


def load_instance(file_path):
    with open(file_path, 'rb') as load_file:
        saved_dataset = pickle.load(load_file)
        return saved_dataset


def save_instance(the_object, file_path):
    # Stores the instance for a multiple classification structure
    with open(file_path, 'wb') as the_file:
        pickle.dump(the_object, the_file)