import pandas as pd

train_path = '../input/train.csv'
train_path_long = '../input/train_long.csv'
test_path       = '../input/test.csv'
test_path_long  = '../input/test_long.csv'


train = pd.read_csv(train_path)
train.rename(index=str, columns={'0': ''})