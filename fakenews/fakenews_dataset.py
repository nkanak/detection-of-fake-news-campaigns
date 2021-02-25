import os
import json
import logging

from datetime import datetime

from models import Tweet, User, BotometerScores


class FakeNewsDataset:
    """The fake news dataset model."""

    def __init__(
        self,
        real_news_retweets_path,
        fake_news_retweets_path,
        user_profiles_path,
        user_followers_path,
    ):
        self._users_by_id = {}
        self._users_by_username = {}
        self._tweets_by_id = {}
        self._real_news_retweets_path = real_news_retweets_path
        self._fake_news_retweets_path = fake_news_retweets_path
        self._user_profiles_path = user_profiles_path
        self._user_followers_path = user_followers_path

    def get_user(self, user_id):
        """Read a user and its followers from disk.
        """
        if user_id in self._users_by_id:
            return self._users_by_id[user_id]

        # load user from file
        with open("{}/{}.json".format(self._user_profiles_path, user_id)) as json_file:
            user_dict = json.load(json_file)
            user = User(user_id)
            self._users_by_id[user_id] = user

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

            user.following_count = user_dict["friends_count"]

        # load user followers from file
        with open("{}/{}.json".format(self._user_followers_path, user_id)) as json_file:
            followers_dict = json.load(json_file)

            if "followers" in followers_dict:
                for follower_id in followers_dict["followers"]:
                    user.followers.add(str(follower_id))

        return user

    def load(self): 
        """Load the dataset.
        """
        for fentry in os.scandir(self._fake_news_retweets_path):
            if fentry.is_dir():
                retweets_path = "{}/retweets".format(fentry.path)
                if os.path.isdir(retweets_path):
                    self._load_retweets_from_disk(retweets_path, real=False)

        for fentry in os.scandir(self._real_news_retweets_path):
            if fentry.is_dir():
                retweets_path = "{}/retweets".format(fentry.path)
                if os.path.isdir(retweets_path):
                    self._load_retweets_from_disk(retweets_path, real=True)

    def _load_retweets_from_disk(self, dirpath, real):
        for fentry in os.scandir(dirpath):
            if fentry.path.endswith(".json") and fentry.is_file():
                with open(fentry.path) as json_file:
                    retweets_dict = json.load(json_file)
                    retweets = retweets_dict.get("retweets", [])
                    if len(retweets) == 0:
                        continue
                    logging.info("Loading retweets from file {}".format(fentry.path))
                    self._load_retweets(retweets, real=real)

    def _load_retweets(self, retweets, real):
        for retweet in retweets: 
            self._update_tweet(retweet)

        # TODO
        # Handle retweet
        pass

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

        #user_id = str(tweet_dict["userid"])
        #user = self._get_user(user_id)
        #tweet.user = user

        #screenname = tweet_dict.get("userscreenname", None)
        #if screenname is not None:
        #    user.screenname = screenname
        #    self._users_by_username[screenname] = user

        return tweet

    @property
    def users_by_id(self):
        return self._users_by_id

    @property
    def tweets_by_id(self):
        return self._tweets_by_id

    def __repr__(self):
        return "FakeNewsDataset(%r, %r, %r)" % (
            self._users_by_id,
            self._users_by_username,
            self._tweets_by_id,
        )

