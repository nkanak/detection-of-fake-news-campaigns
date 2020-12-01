
from datetime import datetime

import jgrapht

from models import (
    Tweet,
    User,
    Dataset,
)


def _find_retweet_source(dataset: Dataset, retweet, previous_retweets):
    """Given a retweet and all previous retweers estimate from which 
    retweet it originated.
    """

    # FIXME
    # Check if text contains RT @ 
    # Check if we follow some of the previous users that retweeted
    # Assign to most popular based on preferential attachment

    pass



def create_dag(dataset: Dataset, tweet_id: str, min_retweets=5): 
    """Given a tweet id, create a dag.
    """
    tweet = dataset.tweets_by_id[tweet_id]
    if tweet.is_retweet: 
        raise ValueError('Dag cannot be created from a retweet')

    if len(tweet.retweeted_by) < min_retweets:
        return None

    print('-------------- new dag -----------------')
    retweets = sorted(
        tweet.retweeted_by, key=lambda t: t.created_at, reverse=True
    )

    dag = jgrapht.create_graph(directed=True, any_hashable=True)
    dag.add_vertex(tweet)

    previous = []
    previous.append(tweet)

    while len(retweets) != 0: 
        cur = retweets.pop()
        cur_retweet_of = _find_retweet_source(dataset, cur, previous)

        # FIXME

        pass

    # FIXME !
    
    return dag


def create_dags(dataset: Dataset, min_retweets=5): 
    """Given a dataset create all dags
    """

    for tweet in dataset.tweets_by_id.values(): 
        if not tweet.is_retweet:
            dag = create_dag(dataset, tweet.id, min_retweets=min_retweets)
            if dag is not None:
                yield dag


