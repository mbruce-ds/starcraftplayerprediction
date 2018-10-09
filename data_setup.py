import pandas as pd

test_path       = '../starting_data/TEST.CSV'
test_path_long  = '../starting_data/TEST_LONG.CSV'
train_path      = '../starting_data/TRAIN.CSV'
train_path_long = '../starting_data/TRAIN_LONG.CSV'



train = pd.read_csv(train_path, sep='|', names=['temp'])
train = train.temp.str.split(',', expand=True)
