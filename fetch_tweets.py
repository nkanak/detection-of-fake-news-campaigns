import argparse
import json
import time
import os
import requests

parser = argparse.ArgumentParser(epilog='Example: python fetch_tweets.py --bearer-token your_token --query your_url_encoded_query')
parser.add_argument('--bearer-token', help='The Bearer Token for the twitter API', dest='BEARER_TOKEN', type=str, required=True)
parser.add_argument('--query', help='A URL encoded query for the twitter API', dest='QUERY', type=str, required=True)
parser.add_argument('--continue-from-the-last-endpoint', help='Whether to continue or not from the last accessed twitter endpoint', dest='CONTINUE_FROM_THE_LAST_ENDPOINT', action='store_true')
args = parser.parse_args()

BEARER_TOKEN = args.BEARER_TOKEN
QUERY = args.QUERY
CONTINUE_FROM_THE_LAST_ENDPOINT = args.CONTINUE_FROM_THE_LAST_ENDPOINT

headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
BASE_ENDPOINT = 'https://api.twitter.com/1.1/search/tweets.json'
DIRPATH = 'raw_data'

if CONTINUE_FROM_THE_LAST_ENDPOINT:
    with open('last_endpoint.txt') as f:
        endpoint = f.readline()
else:
    endpoint = f'{BASE_ENDPOINT}?q={QUERY}&count=1000'

while True:
    while True:
        res = requests.get(endpoint, headers=headers)
        print(f'{res.url}, {res.status_code}')
        if res.status_code == 200:
            break
        print('Sleeping for 15 minutes!')
        time.sleep(15*60)
    data = res.json()
    existing_tweets = os.listdir(DIRPATH)
    for tweet in data['statuses']:
        filename = f'{tweet["id_str"]}.json'
        if filename in existing_tweets:
            continue
        filepath = f'{DIRPATH}/{filename}'
        with open(filepath, 'w') as f:
           json.dump(tweet, f, indent=2)

    endpoint = BASE_ENDPOINT + data['search_metadata']['next_results']
    with open('last_endpoint.txt', 'w') as f:
        f.write(endpoint)