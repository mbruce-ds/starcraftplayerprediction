import pandas as pd
import numpy as np
import math
import statistics
import time
from collections import Counter

# we'll only be using the more detailed dataset, ignore the smaller one
train_path_long = '../input/train_long.csv'
test_path_long  = '../input/test_long.csv'

train = pd.read_csv(train_path_long, low_memory = False)
train = train.drop(['Unnamed: 0'], axis = 1)
train = train.rename(columns = {'0': 'battleneturl', '1' : 'race'})

test = pd.read_csv(test_path_long, low_memory = False)
test = test.drop(['Unnamed: 0'], axis = 1)
test = test.rename(columns = {'0' : 'race'})

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
# single barcode player who played all three. This player is a random race player.


# anyways, time for some feature engineering
# since the dataset we're given is somewhat gross to work with, we're going to
# extract the features and put them into a new dataframe that we will predict
# with. this way we can also just save the features for later use and we won't
# have to always reprocess the dataset every time we want to test something
X_columns = ['battleneturl', 'playername', 'race', 'gamelength', 'apm', '1_usage', '2_usage', '3_usage', '4_usage', '5_usage', '6_usage', '7_usage', '8_usage', '9_usage', '0_usage', 'chattiness', 'worker_box_count']
X_train = pd.DataFrame(index = train.index, columns = X_columns)

X_test_columns = ['race', 'gamelength', 'apm', '1_usage', '2_usage', '3_usage', '4_usage', '5_usage', '6_usage', '7_usage', '8_usage', '9_usage', '0_usage', 'chattiness', 'worker_box_count']
X_test = pd.DataFrame(index = test.index, columns = X_test_columns)

workers = ['SCV', 'Probe', 'Drone']

for game in train.index:
    start = time.time()
    chat_count = 0
    time_frame = 0
    total_actions = 0
    worker_box_count = 0
    selected = []
    camera_x = []
    camera_y = []
    control_group_count = {'1' : 0, '2' : 0, '3' : 0, '4' : 0, '5' : 0, '6' : 0, '7' : 0, '8' : 0, '9' : 0, '0' : 0}
    control_group = {'1' : [], '2' : [], '3' : [], '4' : [], '5' : [], '6' : [], '7' : [], '8' : [], '9' : [], '0' : []}
    # dicts are way faster to iterate over than using iloc on a dataframe
    actions = train.iloc[game].to_dict()
    url = actions['battleneturl']
    name = actions['playername']
    race = actions['race']
    for i in range(2,26825):
        i_string = str(i)
        if(type(actions[i_string]) is not float):
            action_info = actions[i_string].split(':')
            # failsafe check in case if a chatlog messed up the string splitting...
            # which happens... thanks Golden for typing "gl, hf"
            if(len(action_info) > 1):
                time_frame = int(action_info[0])
                if(action_info[1] == 'ChatEvent'):
                    chat_count += 1
                else:
                    if(action_info[1] == 'ControlGroupEvent'):
                        control_group_count[action_info[2]] += 1
                        if(action_info[3] != 2):
                            control_group[action_info[2]] = selected
                        if(action_info[3] == 2):
                            selected = control_group[action_info[2]]
                    if(action_info[1] == 'SelectionEvent'):
                        selected = action_info[2].split(';')
                        if(len(selected) > 1):
                            if(all([x in workers for x in selected])):
                                worker_box_count += 1
                    total_actions += 1
        else:
            break
    game_length = (time_frame + 1)/ (16*60)
    apm = total_actions / game_length
    control_group_importance = {k : v/game_length for k, v in control_group_count.items()}
    X_train.iloc[game] = [url, name, race, game_length, apm, control_group_importance['1'], control_group_importance['2'], control_group_importance['3'], control_group_importance['4'], control_group_importance['5'], control_group_importance['6'], control_group_importance['7'], control_group_importance['8'], control_group_importance['9'], control_group_importance['0'], chat_count, worker_box_count/min(game_length, 7)]
    end = time.time()
    # print(X.iloc[game], '\n\n\n', 'Processed in: {:.4f} seconds\n'.format(end - start), '-'*50, '\n')

for game in test.index:
    chat_count = 0
    time_frame = 0
    total_actions = 0
    worker_box_count = 0
    selected = []
    camera_x = []
    camera_y = []
    control_group_count = {'1' : 0, '2' : 0, '3' : 0, '4' : 0, '5' : 0, '6' : 0, '7' : 0, '8' : 0, '9' : 0, '0' : 0}
    control_group = {'1' : [], '2' : [], '3' : [], '4' : [], '5' : [], '6' : [], '7' : [], '8' : [], '9' : [], '0' : []}
    # dicts are way faster to iterate over than using iloc on a dataframe
    actions = test.iloc[game].to_dict()
    race = actions['race']
    for i in range(1,14626):
        i_string = str(i)
        if(type(actions[i_string]) is not float):
            action_info = actions[i_string].split(':')
            # failsafe check in case if a chatlog messed up the string splitting...
            # which happens... thanks Golden for typing "gl, hf"
            if(len(action_info) > 1):
                time_frame = int(action_info[0])
                if(action_info[1] == 'ChatEvent'):
                    chat_count += 1
                else:
                    if(action_info[1] == 'ControlGroupEvent'):
                        control_group_count[action_info[2]] += 1
                        if(action_info[3] != 2):
                            control_group[action_info[2]] = selected
                        if(action_info[3] == 2):
                            selected = control_group[action_info[2]]
                    if(action_info[1] == 'SelectionEvent'):
                        selected = action_info[2].split(';')
                        if(len(selected) > 1):
                            if(all([x in workers for x in selected])):
                                worker_box_count += 1
                    total_actions += 1
        else:
            break
    game_length = (time_frame + 1)/ (16*60)
    apm = total_actions / game_length
    control_group_importance = {k : v/game_length for k, v in control_group_count.items()}
    X_test.iloc[game] = [race, game_length, apm, control_group_importance['1'], control_group_importance['2'], control_group_importance['3'], control_group_importance['4'], control_group_importance['5'], control_group_importance['6'], control_group_importance['7'], control_group_importance['8'], control_group_importance['9'], control_group_importance['0'], chat_count, worker_box_count/min(game_length, 7)]

train_features_path = 'features_train.csv'
test_features_path = 'features_test.csv'
X_train.to_csv(train_features_path, index = False)
X_test.to_csv(test_features_path, index = False)
