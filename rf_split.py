import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
import warnings
from sklearn.utils import shuffle
warnings.filterwarnings('always')

features = pd.read_csv("C:/Users/Ameer/Documents/sdp/data/mozilla_v1.csv")
features = shuffle(features)
x, y = features.iloc[:,:-1],features.iloc[:,-1]


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.3, 
                                                    random_state=123)
rf = RandomForestClassifier(n_estimators=70,oob_score=True,n_jobs=1,random_state=101,max_features=None, min_samples_leaf=30)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

best_preds = np.asarray([np.argmax(line) for line in y_pred])

print ("Numpy array accuracy: ", accuracy_score(y_test, best_preds))
print ("Numpy array precision: ", precision_score(y_test, best_preds, average='weighted',zero_division=1))
print ("Numpy array recall: ", recall_score(y_test, best_preds, average='weighted'))
print ("Numpy array f-measure: ", f1_score(y_test, best_preds, average='weighted', labels=np.unique(best_preds)))

confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)

