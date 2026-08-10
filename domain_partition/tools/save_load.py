import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)


def load_object(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

