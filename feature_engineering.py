import pandas as pd
import numpy as np
import math
import time
from collections import Counter
pd.set_option('display.max_colwidth', 500)

# we'll only be using the more detailed dataset, ignore the smaller one
train_path_long = '../input/train_long.csv'
test_path_long  = '../input/test_long.csv'

train = pd.read_csv(train_path_long, low_memory=False)
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


# anyways, time for some feature engineering
# since the dataset we're given is somewhat gross to work with, we're going to
# extract the features and put them into a new dataframe that we will predict
# with
X_columns = ['battleneturl', 'playername', 'race', '1_usage', '2_usage', '3_usage', '4_usage', '5_usage', '6_usage', '7_usage', '8_usage', '9_usage', '0_usage', 'chattiness']
X = pd.DataFrame(index = train.index, columns=X_columns)

for game in train.index:
    start = time.time()
    action_id = 2
    current_count = 0
    chat_count = 0
    control_group_count = {'1' : 0, '2' : 0, '3' : 0, '4' : 0, '5' : 0, '6' : 0, '7' : 0, '8' : 0, '9' : 0, '0' : 0}
    while(type(train.iloc[game].iloc[action_id]) is not float):
        action_info = train.iloc[game].iloc[action_id].split(':')
        if(len(action_info) > 1):
            if(action_info[1] == 'ControlGroupEvent'):
                control_group_count[action_info[2]] += 1
            if(action_info[1] == 'ChatEvent'):
                action_id += 1
                chat_count += 1
        action_id += 1
    control_group_importance = {k : math.log1p(v) for k, v in control_group_count.items()}
    X.iloc[game] = [train.iloc[game].battleneturl, train.iloc[game].playername, train.iloc[game].race, control_group_importance['1'], control_group_importance['2'], control_group_importance['3'], control_group_importance['4'], control_group_importance['5'], control_group_importance['6'], control_group_importance['7'], control_group_importance['8'], control_group_importance['9'], control_group_importance['0'], chat_count]
    end = time.time()
    print(X.iloc[game], '\n\n\n', 'Processed in: {:.2f} seconds\n'.format(end - start), '-'*50, '\n')
