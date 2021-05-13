
from datetime import datetime

import re
import random
import jgrapht

from models import (
    Tweet,
    User,
)

def _lookup_RT(text): 
    match = re.search(r'RT\s@((\w){1,15}):', text)
    if match: 
        return match.group(1)
    return None


def _find_retweet_source(dataset, retweet, previous_retweets):
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


def create_dag(dataset, tweet_id: str, min_retweets=5): 
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
    dag.vertex_attrs[tweet]['delay'] = 0

    previous = []
    previous.append(tweet)

    while len(retweets) != 0: 
        cur = retweets.pop()
        dag.add_vertex(vertex=cur)

        cur_retweet_of = _find_retweet_source(dataset, cur, previous)
        dag.add_edge(cur, cur_retweet_of)

        dag.vertex_attrs[cur]['delay'] = abs((cur.created_at-cur_retweet_of.created_at).total_seconds())

        previous.append(cur)

    if tweet.real: 
        dag.graph_attrs['label'] = "real"
    else:
        dag.graph_attrs['label'] = "fake"       

    return dag


def postprocess_dag(dataset, dag, only_user_ids=False): 
    """Given a dag convert vertices to integers and compute features.
    """
    p_dag = jgrapht.create_graph(directed=True, any_hashable=True)

    vid = 0
    tweet_to_id = {}
    for tweet in dag.vertices: 
        p_dag.add_vertex(vertex=vid)

        if only_user_ids: 
            if tweet.user is not None:
                p_dag.vertex_attrs[vid]['user_id'] = tweet.user.id
        else:
            p_dag.vertex_attrs[vid]['delay'] = dag.vertex_attrs[tweet]['delay']
            p_dag.vertex_attrs[vid]['followers_count'] = max(len(tweet.user.followers), tweet.user.followers_count)
            p_dag.vertex_attrs[vid]['following_count'] =  max(len(tweet.user.following), tweet.user.following_count)

            for key in ['verified', 'protected', 'favourites_count', 'listed_count', 'statuses_count']:
                p_dag.vertex_attrs[vid][key] = int(getattr(tweet.user, key))

            if tweet.user.embedding is not None:
                p_dag.vertex_attrs[vid]['user_profile_embedding'] = tweet.user.embedding

        tweet_to_id[tweet] = vid
        vid += 1
        
    for e in dag.edges: 
        u = dag.edge_source(e)
        v = dag.edge_target(e)

        p_dag.add_edge(tweet_to_id[u], tweet_to_id[v])

    p_dag.graph_attrs['label'] = dag.graph_attrs['label']

    return p_dag



def create_dags(dataset, min_retweets=5, only_user_ids=False): 
    """Given a dataset create all dags
    """

    for tweet in dataset.tweets_by_id.values(): 
        if not tweet.is_retweet:
            dag = create_dag(dataset, tweet.id, min_retweets=min_retweets)
            if dag is not None:
                yield postprocess_dag(dataset, dag, only_user_ids=only_user_ids)



def dag_to_json(dag):
    out = "{"
    out += "\"label\":\"{}\"".format(dag.graph_attrs['label'])
    out += ",\"nodes\":["
    first = True
    for v in dag.vertices:
        if first: 
            first = False
        else:
            out += ","
        out += "{"    
        out += "\"id\":\"{}\"".format(v)
        for k, v in dag.vertex_attrs[v].items():
            out += ",\"{}\":\"{}\"".format(k, v)
        out += "}"        
    out += "]"
    out += ",\"edges\":["
    first = True
    for e in dag.edges:
        if first: 
            first = False
        else:
            out += ","
        out += "{"
        out += "\"source\":\"{}\"".format(dag.edge_source(e))
        out += ",\"target\":\"{}\"".format(dag.edge_target(e))
        out += "}"
    out += "]"
    out += "}"
    return out
