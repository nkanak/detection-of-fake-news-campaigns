#!/usr/bin/env python

#
# Preprocess the FakeNews dataset. Create a folder which contains one file 
# per tweet. Inside each tweet we have its list of retweets.
#

import argparse
import os
import logging
import utils
import random
import copy
import logging

import models
from dataset import FakeNewsDataset
from trees import create_trees, tree_to_dict
import json
from datetime import datetime 

USER_PROFILES_PATH = "../raw_data/user_profiles"
USER_FOLLOWERS_PATH = "../raw_data/user_followers"


def load_user_from_disk(user_id):
    print("Looking for user {}".format(user_id))
    user = models.User(user_id)

    # load user from file
    with open("{}/{}.json".format(USER_PROFILES_PATH, user_id)) as json_file:
        user_dict = json.load(json_file)
        if str(user_dict["id"]) != user_id:
            raise ValueError(
                "Invalid userid {} in json files".format(str(user_dict["id"]))
            )

        for key in [
            "followers_count",
            "listed_count",
            "favourites_count",
            "statuses_count",
        ]:
            current_value = getattr(user, key)
            if current_value is None or current_value == 0:
                setattr(user, key, user_dict.get(key, 0))

        for key in [
            "verified",
            "protected",
        ]:
            current_value = getattr(user, key)
            if current_value is None or current_value is False:
                setattr(user, key, user_dict.get(key, False))

        if user.following_count is None or user.following_count == 0:
            user.following_count = user_dict.get("friends_count", 0)

        if user.description is None:
            user.description = user_dict.get("description", None)

    # load user followers from file
    with open("{}/{}.json".format(USER_FOLLOWERS_PATH, user_id)) as json_file:
        followers_dict = json.load(json_file)
        for follower_id in followers_dict.get("followers", []):
            user.followers.add(str(follower_id))

    return user


def create_tweet(tweet_dict, real):
    tweet = models.Tweet(str(tweet_dict["id"]))
    tweet.real = real
    tweet.created_at = datetime.strptime(
            tweet_dict["created_at"], "%a %b %d %H:%M:%S %z %Y"
        )
    tweet.text = tweet_dict["text"]
    
    return tweet

def get_user_id(tweet_dict):
    if "user" in tweet_dict:
        user_id = str(tweet_dict["user"]["id"])
    elif "userid" in tweet_dict:
        user_id = str(tweet_dict["userid"])
    else:
        raise ValueError("Failed to parse user in tweet: {}".format(tweet_dict))
    return user_id

def create_tree(tweet_dict): 
    real = tweet_dict["label"] == "real"
    main_tweet = create_tweet(tweet_dict, real=real)
    main_tweet.user = load_user_from_disk(get_user_id(tweet_dict))

    for retweet_dict in tweet_dict["retweets"]:
        retweet = create_tweet(retweet_dict, real=real)
        retweet.retweet_of = main_tweet
        main_tweet.retweeted_by.append(retweet)
        retweet.user = load_user_from_disk(get_user_id(retweet_dict))

    # TODO: add code for tree create
    # TODO: handle missing user profiles


def run(args):

    random.seed(31)

    logging.info("Creating trees")
    output_path = "trees2"
    os.makedirs(output_path, exist_ok=True)

    count = 0
    for fentry in os.scandir(args.tweets):
        tweet_path = fentry.path
        if count % 250 == 0: 
            logging.info("{}".format(count))
        count += 1        
        with open(tweet_path) as json_file:
            tweet_dict = json.load(json_file)
            dag = create_tree(tweet_dict)

            #logging.debug("Loading retweets from file {}".format(path))


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)-15s %(name)-15s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(epilog="Example: python create_trees.py")
    parser.add_argument(
        "--tweets",
        help="Tweets directory",
        dest="tweets",
        type=str,
        default="tweets1"
    )

    args = parser.parse_args()
    run(args)
