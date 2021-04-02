import os
import json
import pickle
import numpy as np
from typing import List, Dict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def collect_user_ids(dirpath):
    """

    :param dirpath:
    :return:
    """
    user_ids = set()
    for fentry in os.scandir(dirpath):
        if fentry.path.endswith(".json") and fentry.is_file():
            with open(fentry.path) as json_file:
                full_tweet = json.load(json_file)
                user_ids.add(full_tweet["user"]["id_str"])
                retweeted_status = full_tweet.get("retweeted_status")
                if retweeted_status is not None:
                    user_ids.add(retweeted_status["user"]["id_str"])
    return list(user_ids)


def write_object_to_pickle_file(path, obj):
    """

    :param path:
    :param obj:
    :return:
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def read_pickle_from_file(path):
    """

    :param path:
    :return:
    """
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def load_glove_embeddings(filepath: str) -> Dict:
    embeddings_index = {}
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    return embeddings_index

def generate_tokens_from_text(text:str, lowercased:bool=True, stopwords_removed:bool=True) -> List[str]:
    """Convert a text to a list of tokens.
    """
    if lowercased:
        text = text.lower()
    tokens = word_tokenize(text)
    if stopwords_removed:
        tokens = [w for w in tokens if not w in stop_words]
    return tokens