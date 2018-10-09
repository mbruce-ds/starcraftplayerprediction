# This script expands the csv rows to all be of the same length, so that we can easily read it from a csv into a data
# frame next usage

import pandas as pd

train_path = '../starting_data/TRAIN.CSV'
train_path_long = '../starting_data/TRAIN_LONG.CSV'
test_path       = '../starting_data/TEST.CSV'
test_path_long  = '../starting_data/TEST_LONG.CSV'

train_write_path = '../input/train.csv'
train_write_path_long = '../input/train_long.csv'
test_write_path       = '../input/test.csv'
test_write_path_long  = '../input/test_long.csv'

train = pd.read_csv(train_path, sep='|', names=['temp'])
train = train.temp.str.split(',', expand=True)
train.to_csv(train_write_path)

train_long = pd.read_csv(train_path_long, sep='|', names=['temp'])
train_long = train_long.temp.str.split(',', expand=True)
train_long.to_csv(train_write_path_long)

test = pd.read_csv(test_path, sep='|', names=['temp'])
test = test.temp.str.split(',', expand=True)
test.to_csv(test_write_path)

test_long = pd.read_csv(test_path_long, sep='|', names=['temp'])
test_long = test_long.temp.str.split(',', expand=True)
test_long.to_csv(test_write_path_long)