import json
import utils
import botometer
import argparse
from tqdm import tqdm
from tweepy import TweepError
import time

def run(args):
    rapidapi_key = args.rapidapi_key
    twitter_app_auth = {
        'consumer_key': args.consumer_key,
        'consumer_secret': args.consumer_key_secret,
        'access_token': args.access_token,
        'access_token_secret': args.access_token_secret,
    }
    bom = botometer.Botometer(wait_on_ratelimit=True,
                              rapidapi_key=rapidapi_key,
                              **twitter_app_auth)

    if args.continue_from_the_last_user:
        print("Continue from the last user!")
        accounts = utils.read_pickle_from_file("remaining_users_botometer.pkl")
    else:
        accounts = utils.collect_user_ids(args.input_dir)
    print("Number of user accounts:", len(accounts))
    pbar = tqdm(total=len(accounts))
    while len(accounts) != 0:
        utils.write_object_to_pickle_file("remaining_users_botometer.pkl", accounts)
        user_id = accounts[0]
        pbar.set_postfix_str(f"User id: {user_id}")
        try:
            result = bom.check_account(user_id)
        except TweepError as err:
            response = err.response
            if response.status_code == 401:
                result = {
                    "user_id": user_id,
                    "error_code": 2,
                    "status_code": 401,
                    "error_message": "Not authorized or Account suspended."
                }
            else:
                pbar.set_postfix_str(f"{err} {response.status_code}, Sleeping for 5 minutes!")
                time.sleep(60*5)
                continue
        except Exception as err:
            print(err)
            break
        with open(f"{args.output_dir}/{user_id}.json", "w") as f:
            json.dump(result, f, indent=2)
        accounts.pop(0)
        pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch further information about each user using the Botometer (BotOrNot) API",
        epilog="Example: fetch_botometer_data.py --input-directory input_dirpath --rapidapi-key rapidapi_key" +
               "--consumer-key twitter_api_key --consumer-key-secret twitter_api_key_secret" +
               "--access-token twitter_access_token --access-token-secret twitter_access_token_secret"

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
        default="raw_data_botometer",
    )
    parser.add_argument(
        "--rapidapi-key",
        help="The Bearer Token for the twitter API",
        dest="rapidapi_key",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--consumer-key",
        help="The Bearer Token for the twitter API",
        dest="consumer_key",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--consumer-key-secret",
        help="The Bearer Token for the twitter API",
        dest="consumer_key_secret",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--access-token",
        help="The Bearer Token for the twitter API",
        dest="access_token",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--access-token-secret",
        help="The Bearer Token for the twitter API",
        dest="access_token_secret",
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
