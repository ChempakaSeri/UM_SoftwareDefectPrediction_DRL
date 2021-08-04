import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
import warnings
from sklearn.utils import shuffle
warnings.filterwarnings('always')

features = pd.read_csv("C:/Users/Ameer/Documents/sdp/data/postgres_v1.csv")
features = shuffle(features)
x, y = features.iloc[:,:-1],features.iloc[:,-1]


x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3)
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

best_preds = np.asarray([np.argmax(line) for line in y_pred])


print ("Numpy array precision: ", precision_score(y_test, best_preds, average='weighted',zero_division=1))
#   UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
#  _warn_prf(average, modifier, msg_start, len(result))
print ("Numpy array accuracy: ", accuracy_score(y_test, best_preds))
print ("Numpy array recall: ", recall_score(y_test, best_preds, average='weighted'))
print ("Numpy array f-measure: ", f1_score(y_test, best_preds, average='weighted', labels=np.unique(best_preds)))

confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)

