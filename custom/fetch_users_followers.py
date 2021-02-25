import argparse
import os
import json
import requests
import time
import pickle


def collect_user_ids(dirpath):
    # Create a set of users.
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


def run(args):
    BASE_ENDPOINT = "https://api.twitter.com/1.1/followers/ids.json"

    if args.continue_from_the_last_user:
        print("Continue from the last user!")
        with open("remaining_user_ids_followers.pkl", "rb") as f:
            user_ids = pickle.load(f)
    else:
        user_ids = collect_user_ids(args.input_dir)

    # Start fetching and saving data for each user.
    headers = {"Authorization": f"Bearer {args.bearer_token}"}
    print(f"Number of user ids {len(user_ids)}")
    while len(user_ids) != 0:
        with open("remaining_user_ids_followers.pkl", "wb") as f:
            pickle.dump(user_ids, f)
        user_id = user_ids.pop(0)
        next_cursor = -1
        user_followers = []
        while True:
            payload = {
                "user_id": user_id,
                "cursor": next_cursor,
                "count": 5000,
            }
            while True:
                res = requests.get(BASE_ENDPOINT, headers=headers, params=payload)
                print(f"{res.url}, {res.status_code}")
                if res.status_code == 404:
                    print("User not found!")
                    break
                if res.status_code == 401:
                    print("Not authorized or Account suspended.")
                    break
                if res.status_code == 200:
                    break
                print("Sleeping for 15 minutes!")
                time.sleep(15 * 60)
            if res.status_code == 404:
                break
            if res.status_code == 401:
                break
            data = res.json()
            user_followers.extend(data["ids"])
            next_cursor = data["next_cursor"]
            # The user does not have other followers.
            if next_cursor == 0:
                break
        filepath = f"{args.output_dir}/{user_id}.json"
        if res.status_code == 404:
            payload = {
                "user_id": user_id,
                "error_code": 1,
                "status_code": 404,
                "error_message": "The user does not exist anymore!",
            }
        elif res.status_code == 401:
            payload = {
                "user_id": user_id,
                "error_code": 2,
                "status_code": 401,
                "error_message": "Not authorized or Account suspended.",
            }
        else:
            payload = {
                "user_id": user_id,
                "followers": user_followers,
            }
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch the followers of each user for a list of tweets.",
        epilog="Example: python fetch_users_followers.py --input-directory input_dirpath" +
               "--output-directory output_dirpath --bearer-token your_bearer_token",
    )
    parser.add_argument(
        "--input-directory",
        help="Input directory containing tweet files in json format",
        dest="input_dir",
        type=str,
    )
    parser.add_argument(
        "--output-directory",
        help="Output directory containing user files in json format",
        dest="output_dir",
        type=str,
        default="raw_data_followers",
    )
    parser.add_argument(
        "--bearer-token",
        help="The Bearer Token for the twitter API",
        dest="bearer_token",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--continue-from-the-last-user",
        help="Continue from the last user",
        dest="continue_from_the_last_user",
        action="store_true"
    )
    args = parser.parse_args()

    run(args)
