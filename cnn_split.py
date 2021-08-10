# Import `train_test_split` from `sklearn.model_selection`
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.utils import to_categorical
import warnings

warnings.filterwarnings('ignore')
features = pd.read_csv("C:/Users/Ameer/Documents/sdp/data/bugzilla_v1.csv")
features = shuffle(features)
x, y = features.iloc[:,:-1],features.iloc[:,-1]

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)



y_train = to_categorical(y_train, num_classes=2).astype(np.int64)
y_test = to_categorical(y_test, num_classes=2).astype(np.int64)

# Import `Sequential` from `keras.models`
from keras.models import Sequential
# Import `Dense` from `keras.layers`
from keras.layers import Dense

# Initialize the constructor
model = Sequential()
# Add an input layer 
model.add(Dense(12, activation='relu', input_shape=(14,)))
# Add one hidden layer 
model.add(Dense(8, activation='relu'))
# Add an output layer 
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
history = model.fit(X_train, y_train,epochs=10, batch_size=1, verbose=1)

y_pred = model.predict(X_test)
score = model.evaluate(X_test, y_test,verbose=1)
y_pred=np.argmax(y_pred, axis=1)
y_test=np.argmax(y_test, axis=1)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

## Performance measure
print('\nWeighted Accuracy: '+ str(accuracy_score(y_true=y_test, y_pred=y_pred)))
print('Weighted precision: '+ str(precision_score(y_true=y_test, y_pred=y_pred, average='weighted')))
print('Weighted recall: '+ str(recall_score(y_true=y_test, y_pred=y_pred, average='weighted')))
print('Weighted f-measure: '+ str(f1_score(y_true=y_test, y_pred=y_pred, average='weighted')))

from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(y_true=y_test, y_pred=y_pred, target_names=['Class 1', 'Class 2']))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.plot(epochs_range, acc, 'bo', label='Training acc')
plt.plot(epochs_range, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()