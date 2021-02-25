import os
import json
import pickle


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

