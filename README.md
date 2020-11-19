# Astroturfing project

## Setup
```
pip install -r requirements.txt
```

## Retrieve tweets from twitter
Example: `python fetch_tweets.py --bearer-token your_token --query your_url_encoded_query`

```
usage: fetch_tweets.py [-h] --bearer-token BEARER_TOKEN --query QUERY [--continue-from-the-last-endpoint]

optional arguments:
  -h, --help            show this help message and exit
  --bearer-token BEARER_TOKEN
                        The Bearer Token for the twitter API
  --query QUERY         A URL encoded query for the twitter API
  --continue-from-the-last-endpoint
                        Whether to continue or not from the last accessed twitter endpoint

Example: python fetch_tweets.py --bearer-token your_token --query your_url_encoded_query
```

## Train a Graph Neural Network
```
# TODO
``` 