import json
import os
import pandas as pd

# %% Create dataframe


def create_dataframe(account_data_list):
    rows = []

    for account_data in account_data_list:
        user_follower_count = account_data["userFollowerCount"]
        user_following_count = account_data["userFollowingCount"]
        follower_following_ratio = user_follower_count / max(1, user_following_count)

        row = {
            "user_media_count": account_data["userMediaCount"],
            "user_follower_count": account_data["userFollowerCount"],
            "user_following_count": account_data["userFollowingCount"],
            "user_has_profil_pic": account_data["userHasProfilPic"],
            "user_is_private": account_data["userIsPrivate"],
            "follower_following_ratio": follower_following_ratio,
            "user_biography_length": account_data["userBiographyLength"],
            "username_length": account_data["usernameLength"],
            "username_digit_count": account_data["usernameDigitCount"],
            "is_fake": account_data["isFake"],
        }
        rows.append(row)

    dataframe = pd.DataFrame(rows)
    return dataframe


# %% Import fake/real account data


def import_data(dataset_path, dataset_version):
    fake_account_path = os.path.join(
        dataset_path, dataset_version, "fakeAccountData.json"
    )
    real_account_path = os.path.join(
        dataset_path, dataset_version, "realAccountData.json"
    )

    with open(fake_account_path) as json_file:
        fake_account_data = json.load(json_file)
    with open(real_account_path) as json_file:
        real_account_data = json.load(json_file)

    fake_account_dataframe = create_dataframe(fake_account_data)
    real_account_dataframe = create_dataframe(real_account_data)
    merged_dataframe = pd.concat(
        [fake_account_dataframe, real_account_dataframe], ignore_index=True
    )
    data = dict({"dataset_type": "fake", "dataframe": merged_dataframe})

    return data
