# attempted features that didn't work (lowered score):
# finding apm for each minute in the first 10 minutes
# counting number of times workers were added to control groups
# standard deviation of camera movements


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

train_features_path = 'features_train.csv'
test_features_path = 'features_test.csv'
train_dataset = pd.read_csv(train_features_path)
test_dataset = pd.read_csv(test_features_path)
train_dataset = pd.get_dummies(train_dataset, columns = ['race'])
test_dataset = pd.get_dummies(test_dataset, columns = ['race'])
y = train_dataset.battleneturl
X = train_dataset.drop(['battleneturl', 'playername'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_train, y_train)
print(rfc.score(X_test, y_test))


fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(train_dataset.corr(), annot=True, fmt='.2f', ax = ax, linewidths=0.5, cmap='viridis')
fig.savefig('correlations.png')
plt.close(fig)

full_rfc = RandomForestClassifier(n_estimators = 100)
full_rfc.fit(X, y)
predictions = full_rfc.predict(test_dataset)
prediction_frame = pd.DataFrame({'RowId' : range(1, 341), 'prediction' : predictions})
write_file = 'submission.csv'
prediction_frame.to_csv(write_file, index = False)
