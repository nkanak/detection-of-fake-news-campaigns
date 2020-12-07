# Astroturfing project

## Setup
```
pip install -r requirements.txt
```

## Steps
* Retrieve tweets from twitter
* Retrieve the followers of each user for a list of tweets
* Retrieve the friends of each user for a list of tweets
* Retrieve further information about each user using the Botometer (BotOrNot) API
* Convert tweets to dags
* ...

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

## Retrieve the followers of each user for a list of tweets
Example: `python fetch_users_followers.py --input-directory input_dirpath--output-directory output_dirpath --bearer-token your_bearer_token`

```
usage: fetch_users_followers.py [-h] --input-directory INPUT_DIR [--output-directory OUTPUT_DIR] --bearer-token BEARER_TOKEN [--continue-from-the-last-user]

optional arguments:
  -h, --help            show this help message and exit
  --input-directory INPUT_DIR
                        Input directory containing tweet files in json format
  --output-directory OUTPUT_DIR
                        Output directory containing user files in json format
  --bearer-token BEARER_TOKEN
                        The Bearer Token for the twitter API
  --continue-from-the-last-user
                        Continue from the last user

Example: python fetch_users_followers.py --input-directory input_dirpath--output-directory output_dirpath --bearer-token your_bearer_token
```

## Train a Graph Neural Network
```
# TODO
``` 