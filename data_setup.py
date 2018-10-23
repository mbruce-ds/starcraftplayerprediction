# This script expands the csv rows to all be of the same length, so that we can
# easily read it from a csv into a data frame next usage.

import pandas as pd
import csv

train_path = '../starting_data/TRAIN.CSV'
train_path_long = '../starting_data/TRAIN_LONG.CSV'
test_path       = '../starting_data/TEST.CSV'
test_path_long  = '../starting_data/TEST_LONG.CSV'

train_write_path = '../input/train.csv'
train_write_path_long = '../input/train_long.csv'
test_write_path       = '../input/test.csv'
test_write_path_long  = '../input/test_long.csv'

def max_len_rows(stream):
    max_length = 0
    for rows in csv.reader(stream):
        foo = len(rows)
        if foo > max_length:
            max_length = foo
    return max_length

train = pd.read_csv(train_path, sep = '|', names = ['temp'])
train = train.temp.str.split(',', expand=True)
train.to_csv(train_write_path)
print('Wrote to {}'.format(train_write_path))


with open(train_path_long) as f:
    train_long = pd.read_csv(train_path_long, names = range(max_len_rows(f)), low_memory = False)

train_long.to_csv(train_write_path_long)
print('Wrote to {}'.format(train_write_path_long))


test = pd.read_csv(test_path, sep = '|', names = ['temp'])
test = test.temp.str.split(',', expand=True)
test.to_csv(test_write_path)
print('Wrote to {}'.format(test_write_path))


with open(test_path_long) as f:
    test_long = pd.read_csv(test_path_long, names = range(max_len_rows(f)), low_memory = False)

test_long.to_csv(test_write_path_long)
print('Wrote to {}'.format(test_write_path_long))
