import pickle


def save_picke_file(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_file_picke(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

