import pickle

def save_as_pickle(filername:str, obj:object):
    with open(filername, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_from_pickle(filername:str):
    with open(filername, 'rb') as handle:
        return pickle.load(handle)