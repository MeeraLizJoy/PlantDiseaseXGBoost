# src/utils.py
import pickle
import os

def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)