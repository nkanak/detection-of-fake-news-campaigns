import os
import json
import logging

from datetime import datetime

from models import Tweet, User, BotometerScores

TWEET_ATTRS = {
    "id_str": "id",
    "created_at": "created_at",
    "text": "text",
    "lang": "lang",
    "retweet_count": "retweet_count",
}
USER_ATTRS = {"id_str": "userid", "name": "username", "screen_name": "userscreenname"}


def _strip_tweet(dict_tweet):
    tweet = {}
    for k, v in TWEET_ATTRS.items():
        if k in dict_tweet:
            tweet[v] = dict_tweet[k]

    for k, v in USER_ATTRS.items():
        if k in dict_tweet["user"]:
            tweet[v] = dict_tweet["user"][k]

    retweet = None
    if "retweeted_status" in dict_tweet:
        retweet, _ = _strip_tweet(dict_tweet["retweeted_status"])

    return tweet, retweet


class Dataset:
    """A dataset model."""

    def __init__(self):
        self._users_by_id = {}
        self._users_by_username = {}
        self._tweets_by_id = {}

    def load_users(self, path):
        if not os.path.isdir(path):
            logging.warning("Users dir {} not found!".format(path))
            return
        for fentry in os.scandir(path):
            if fentry.path.endswith(".json") and fentry.is_file():
                with open(fentry.path) as json_file:
                    user_dict = json.load(json_file)
                    user = self._get_user(user_dict["id_str"])

                    for key in [
                        "followers_count",
                        "listed_count",
                        "favourites_count",
                        "statuses_count",
                        "description",
                        "verified",
                        "protected",
                    ]:
                        setattr(user, key, user_dict[key])

                    user.following_count = user_dict['friends_count']


    def load_followers(self, path):
        if not os.path.isdir(path):
            logging.warning("Followers dir {} not found!".format(path))
            return
        for fentry in os.scandir(path):
            if fentry.path.endswith(".json") and fentry.is_file():
                with open(fentry.path) as json_file:
                    followers_dict = json.load(json_file)
                    user = self._get_user(str(followers_dict["user_id"]))

                    if "followers" in followers_dict:
                        for follower_id in followers_dict["followers"]:
                            user.followers.add(str(follower_id))

    def load_tweets(self, path):
        if not os.path.isdir(path):
            logging.warning("Tweets dir {} not found!".format(path))
            return
        for fentry in os.scandir(path):
            if fentry.path.endswith(".json") and fentry.is_file():
                with open(fentry.path) as json_file:
                    full_tweet = json.load(json_file)
                    tweet_dict, retweet_dict = _strip_tweet(full_tweet)

                    tweet = self._update_tweet(tweet_dict)

                    if retweet_dict is not None:
                        retweet = self._update_tweet(retweet_dict)
                        tweet.retweet_of = retweet
                        retweet.retweeted_by.append(tweet)

    def load_botometer(self, path):
        if not os.path.isdir(path):
            logging.warning("Botometer dir {} not found!".format(path))
        for fentry in os.scandir(path):
            if fentry.path.endswith(".json") and fentry.is_file():
                with open(fentry.path) as json_file:
                    boto = json.load(json_file)
                    result = self._parse_botometer(boto)
                    if result is not None:
                        user = self._get_user(result[0])
                        user.botometer_scores = result[1]

    @property
    def users_by_id(self):
        return self._users_by_id

    @property
    def tweets_by_id(self):
        return self._tweets_by_id

    @property
    def users_by_username(self):
        return self._users_by_username

    def _get_user(self, user_id):
        if user_id in self._users_by_id:
            user = self._users_by_id[user_id]
        else:
            user = User(user_id)
            self._users_by_id[user_id] = user
        return user

    def _update_tweet(self, tweet_dict):
        tweet_id = str(tweet_dict["id"])
        if tweet_id in self._tweets_by_id:
            tweet = self._tweets_by_id[tweet_id]
        else:
            tweet = Tweet(tweet_id)
            self._tweets_by_id[tweet_id] = tweet

        created_at_str = tweet_dict["created_at"]
        tweet.created_at = datetime.strptime(created_at_str, "%a %b %d %H:%M:%S %z %Y")
        tweet.text = tweet_dict["text"]

        user_id = str(tweet_dict["userid"])
        user = self._get_user(user_id)
        tweet.user = user

        screenname = tweet_dict.get("userscreenname", None)
        if screenname is not None:
            user.screenname = screenname
            self._users_by_username[screenname] = user

        return tweet

    def _parse_botometer(self, data):
        scores = BotometerScores()

        if "user" not in data:
            # Skip on error
            return None

        user_id = data["user"]["user_data"]["id_str"]
        for key in [
            "astroturf",
            "fake_follower",
            "financial",
            "other",
            "overall",
            "self_declared",
            "spammer",
        ]:
            setattr(scores, key, data["raw_scores"]["english"][key])

        return user_id, scores

    def __repr__(self):
        return "Dataset(%r, %r, %r)" % (
            self._users_by_id,
            self._users_by_username,
            self._tweets_by_id,
        )
