import pandas as pd
import numpy as np
from collections import Counter
pd.set_option('display.max_colwidth', 500)

train_path = '../input/train.csv'
train_path_long = '../input/train_long.csv'
test_path       = '../input/test.csv'
test_path_long  = '../input/test_long.csv'

train = pd.read_csv(train_path, low_memory=False)
train = train.drop(['Unnamed: 0'], axis=1)
train = train.rename(columns={'0': 'battleneturl', '1' : 'race'})

# creates variable with the player name using the url, just for ease of reading
train['playername'] = train['battleneturl'].str.split('/').str[-2]

# counts number of games recorded for each player
game_counts = {}
for name in train.playername.unique():
    game_counts[name] = train['playername'].str.contains(name).value_counts()[True]

# note that either there are no Random players in the data set, or the
# dataset automatically selects the race that Random players roll:
random_players = np.sum(train['race'].str.contains('Random'))

# random_players is 0, meaning no players are marked as random.

game_counts_race = {}
for name in train.playername.unique():
    player_games = train.index[train['playername'] == name].tolist()
    # we use a weird separator prefix to make sure no one is using it in their name...
    name_terran_str  = name + '^&_terran'
    name_protoss_str = name + '^&_protoss'
    name_zerg_str    = name + '^&_zerg'
    game_counts_race[name_terran_str]  = np.sum(train['race'].iloc[player_games].str.contains('Terran'))
    game_counts_race[name_protoss_str] = np.sum(train['race'].iloc[player_games].str.contains('Protoss'))
    game_counts_race[name_zerg_str]    = np.sum(train['race'].iloc[player_games].str.contains('Zerg'))

game_counts_race = {k:v for k, v in game_counts_race.items() if v}

name_split_race = list(game_counts_race.keys())
for i in range(len(name_split_race)):
    name_split_race[i] = name_split_race[i].split('^&_')[0]

player_races_played = Counter(name_split_race).most_common()

# This shows that every player analyzed only played one race, except for a
# single barcode player who played all three. This player is potentially a
# random race player. I'm undecided about how I'll handle barcodes yet.
# They might be different players than the set we're given, or they may be alts
# of players already analyzed

# anyways, time for some feature engineering!

player_avg_apm = {}
for name in train.playername.unique():
    # first, let's calculate the APM of the players over each 5 second interval
    player_games_index = train.index[train['playername'] == name].tolist()
    player_games = train.iloc[player_games_index]
    apm_dict = {}
    avg_apm = {}
    for game in range(len(player_games_index)):
        action_id = 2
        current_count = 0
        total_actions = []
        while(1):
            while(not pd.isna(player_games.iloc[game].iloc[action_id]) and player_games.iloc[game].iloc[action_id][0] != 't'):
                action_id += 1
                current_count += 1
            total_actions.append(current_count)
            if(pd.isna(player_games.iloc[game].iloc[action_id])):
                break
            action_id += 1
            current_count = 0
        apm = np.array(total_actions)*12
        apm_dict[game] = apm
        avg_apm[game] = np.average(apm)
    player_avg_apm[name] = np.average(np.fromiter(avg_apm.values(), dtype=np.float))
    print(name, player_avg_apm[name])
