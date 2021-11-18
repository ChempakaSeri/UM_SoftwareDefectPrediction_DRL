import pandas as pd
from sklearn.utils import shuffle

features = pd.read_csv("C:/Users/Ameer/Desktop/cuya/epn_v5.csv")
features = shuffle(features)
x, y = features.iloc[:,:-1],features.iloc[:,-1]

from sklearn.model_selection import train_test_split

#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1, stratify=y)

from sklearn.neighbors import KNeighborsClassifier

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 3)

# Fit the classifier to the data
knn.fit(X_train,y_train)

#show first 5 model predictions on the test data
knn.predict(X_test)[0:5]

#check accuracy of our model on the test data
knn.score(X_test, y_test)

#KNN with 5-k fold
 

#Grid Search

from sklearn.model_selection import GridSearchCV

#create new a knn model
knn2 = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}

#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)

#fit model to data
knn_gscv.fit(x, y)

#check top performing n_neighbors value
knn_gscv.best_params_

#check mean score for the top performing value of n_neighbors
knn_gscv.best_score_