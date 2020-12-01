
from datetime import datetime

import re
import random
import jgrapht

from models import (
    Tweet,
    User,
    Dataset,
)

def _lookup_RT(text): 
    match = re.search(r'RT\s@((\w){1,15}):', text)
    if match: 
        return match.group(1)
    return None

def _find_retweet_source(dataset: Dataset, retweet, previous_retweets):
    """Given a retweet and all previous retweers estimate from which 
    retweet it originated.
    """
    user = retweet.user
    rt_username = _lookup_RT(retweet.text)

    # Find a tweet from the RT user
    if rt_username is not None:
        candidates = []
        for rt in previous_retweets:
            if rt.user.screenname == rt_username: 
                candidates.append(rt)

        if len(candidates) != 0:
            return max(candidates, key= lambda k: rt.user.popularity)

    # Check if we follow some of the previous users that retweeted
    candidates = []
    for rt in previous_retweets:
        if user.id in rt.user.followers:
            candidates.append(rt)

    if len(candidates) != 0:
        return max(candidates, key= lambda k: rt.user.popularity)

    # Assign to most popular based on popularity
    weights = [rt.user.popularity for rt in previous_retweets]
    return random.choices(previous_retweets, weights=weights, k=1)[0]


def create_dag(dataset: Dataset, tweet_id: str, min_retweets=5): 
    """Given a tweet id, create a dag.
    """
    tweet = dataset.tweets_by_id[tweet_id]
    if tweet.is_retweet: 
        raise ValueError('Dag cannot be created from a retweet')

    if len(tweet.retweeted_by) < min_retweets:
        return None

    retweets = sorted(
        tweet.retweeted_by, key=lambda t: t.created_at, reverse=True
    )

    dag = jgrapht.create_graph(directed=True, any_hashable=True)
    dag.add_vertex(vertex=tweet)

    previous = []
    previous.append(tweet)

    while len(retweets) != 0: 
        cur = retweets.pop()
        dag.add_vertex(vertex=cur)

        cur_retweet_of = _find_retweet_source(dataset, cur, previous)
        dag.add_edge(cur, cur_retweet_of)

        previous.append(cur)
    
    return dag


def create_dags(dataset: Dataset, min_retweets=5): 
    """Given a dataset create all dags
    """

    for tweet in dataset.tweets_by_id.values(): 
        if not tweet.is_retweet:
            dag = create_dag(dataset, tweet.id, min_retweets=min_retweets)
            if dag is not None:
                yield dag


