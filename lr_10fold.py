from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import pandas as pd

features = pd.read_csv("C:/Users/Ameer/Documents/sdp/data/bugzilla_v1.csv")
x, y = features.iloc[:,:-1],features.iloc[:,-1]

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
model = LogisticRegression(solver='lbfgs', max_iter=1000)

# evaluate model
accuracy = cross_val_score(model, x, y, cv=cv, n_jobs=-1, scoring='accuracy')
precision = cross_val_score(model, x, y, cv=cv, n_jobs=-1, scoring='precision')
recall = cross_val_score(model, x, y, cv=cv, n_jobs=-1, scoring='recall')
f1 = cross_val_score(model, x, y, cv=cv, n_jobs=-1, scoring='f1')
# report performance
#print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

